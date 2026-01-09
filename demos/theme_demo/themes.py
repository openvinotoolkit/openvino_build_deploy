import abc
import os
import sys
from pathlib import Path
from typing import Any
import time

import cv2
import numpy as np
import openvino as ov

from decoder import OpenPoseDecoder
from numpy.lib.stride_tricks import as_strided

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


class Theme(abc.ABC):
    def __init__(self):
        self.model_dir = Path(__file__).parent / "model"
        self.assets_dir = Path(__file__).parent / "assets"

    def __download_model(self, model_name: str, precision: str):
        model_path = self.model_dir / model_name / precision / f"{model_name}"
        models_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(models_dir + model_name + ".bin", model_path.with_suffix(".bin").name, model_path.parent)
        utils.download_file(models_dir + model_name + ".xml", model_path.with_suffix(".xml").name, model_path.parent)

    def _load_model(self, model_name: str, precision: str, device: str):
        model_path = self.model_dir / model_name / precision / f"{model_name}.xml"
        if not model_path.exists():
            self.__download_model(model_name, precision)

        core = ov.Core()
        model = core.read_model(model=model_path)
        compiled_model = core.compile_model(model=model, device_name=device)
        return compiled_model

    def _load_asset(self, asset_name: str):
        asset_path = (self.assets_dir / asset_name).with_suffix(".png")
        return cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)

    def _load_assets(self, assets_names: list[str]):
        assets = {}
        for asset_name in assets_names:
            assets[asset_name] = self._load_asset(asset_name)
        return assets

    @abc.abstractmethod
    def run_inference(self, frame: np.ndarray) -> Any:
        return None

    @abc.abstractmethod
    def draw_results(self, image: np.ndarray, detections: Any) -> np.ndarray:
        return image

    def _calculate_iou_matrix(self, boxes1, boxes2):
        # boxes: (N, 4) arrays
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        x11, y11, w1, h1 = np.split(boxes1, 4, axis=1)
        x12, y12 = x11 + w1, y11 + h1
        x21, y21, w2, h2 = np.split(boxes2, 4, axis=1)
        x22, y22 = x21 + w2, y21 + h2
        xi1 = np.maximum(x11, x21.T)
        yi1 = np.maximum(y11, y21.T)
        xi2 = np.minimum(x12, x22.T)
        yi2 = np.minimum(y12, y22.T)
        inter_area = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area.T - inter_area
        return inter_area / (union_area + 1e-6)


class ChristmasTheme(Theme):
    def __init__(self, device: str = "CPU"):
        super().__init__()
        self.emotion_classes = ["neutral", "happy", "sad", "surprise", "anger"]
        self.emotion_mapping = {"neutral": "Rudolph", "happy": "Cupid", "surprise": "Blitzen", "sad": "Prancer", "anger": "Vixen"}

        self.assets = self._load_assets(["santa_beard", "santa_cap", "reindeer_nose", "reindeer_sunglasses", "reindeer_antlers"])

        self.model_precision = "FP16-INT8"
        self.device = device

        self.face_detection_model = None
        self.face_landmarks_model = None
        self.emotions_recognition_model = None

        self.tracked_faces = {}  # track_id: {'box': np.ndarray, ...}
        self.next_face_id = 0
        self.iou_threshold = 0.5
        self.smoothing_tau = 0.3  # seconds

        self.load_models(device)

    def load_models(self, device: str):
        self.device = device
        # face-detection-0204 is the latest model supported by NPU
        self.face_detection_model = self._load_model("face-detection-0204", self.model_precision, device)
        self.face_landmarks_model = self._load_model("facial-landmarks-35-adas-0002", self.model_precision, device)
        self.emotions_recognition_model = self._load_model("emotions-recognition-retail-0003", self.model_precision, device)

    def run_inference(self, frame: np.ndarray) -> Any:
        boxes = self.__detect_faces(frame)
        landmarks = self.__detect_landmarks(frame, boxes)
        emotions = self.__recognize_emotions(frame, boxes)
        detections = list(zip(boxes, landmarks, emotions))
        return self._smooth_detections(detections)

    def draw_results(self, image: np.ndarray, detections: Any) -> np.ndarray:
        # sort by face size
        detections = list(sorted(detections, key=lambda x: x[0][1][2] * x[0][1][3]))

        if not detections:
            return image

        # others are reindeer
        for (score, box), landmarks, emotion in detections[:-1]:
            self.__draw_reindeer(image, landmarks, box)

            (label_width, label_height), _ = cv2.getTextSize(
                text=self.emotion_mapping[emotion],
                fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                fontScale=box[2] / 150,
                thickness=1)
            point = np.mean(landmarks[:4], axis=0, dtype=np.int32) - [label_width // 2, 2 * label_height]
            cv2.putText(
                img=image,
                text=self.emotion_mapping[emotion],
                org=point,
                fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                fontScale=box[2] / 150,
                color=(0, 0, 196),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # the largest face is santa
        self.__draw_santa(image, detections[-1])

        return image

    def _smooth_detections(self, current_detections):
        now = time.time()
        if not current_detections:
            self.tracked_faces = {}
            return []
        # Prepare current faces
        curr_boxes = [det[0][1] for det in current_detections]
        curr_scores = [det[0][0] for det in current_detections]
        curr_landmarks = [det[1] for det in current_detections]
        curr_emotions = [det[2] if len(det) > 2 else None for det in current_detections]
        curr_boxes_np = np.array(curr_boxes)
        # Prepare tracked faces
        tracked_ids = list(self.tracked_faces.keys())
        tracked_boxes_np = np.array([self.tracked_faces[tid]['box'] for tid in tracked_ids]) if tracked_ids else np.zeros((0,4))
        # IoU matrix
        ious = self._calculate_iou_matrix(tracked_boxes_np, curr_boxes_np)
        # Greedy matching
        matched_tracked = set()
        matched_current = set()
        matches = []
        for t_idx, tid in enumerate(tracked_ids):
            if ious.shape[1] == 0:
                break
            c_idx = np.argmax(ious[t_idx])
            if ious[t_idx, c_idx] >= self.iou_threshold and c_idx not in matched_current:
                matches.append((tid, c_idx))
                matched_tracked.add(tid)
                matched_current.add(c_idx)
        # Update matched tracks
        for tid, c_idx in matches:
            prev = self.tracked_faces[tid]
            dt = now - prev['last_update']
            alpha = np.exp(-dt / self.smoothing_tau)
            new_box = prev['box'] * alpha + curr_boxes_np[c_idx] * (1 - alpha)
            self.tracked_faces[tid].update({
                'box': new_box,
                'score': curr_scores[c_idx],
                'landmarks': curr_landmarks[c_idx],
                'emotion': curr_emotions[c_idx],
                'last_update': now
            })
        # Add new tracks
        for c_idx in range(len(current_detections)):
            if c_idx not in matched_current:
                tid = self.next_face_id
                self.next_face_id += 1
                self.tracked_faces[tid] = {
                    'box': curr_boxes_np[c_idx],
                    'score': curr_scores[c_idx],
                    'landmarks': curr_landmarks[c_idx],
                    'emotion': curr_emotions[c_idx],
                    'last_update': now
                }
        # Remove old tracks
        for tid in tracked_ids:
            if tid not in matched_tracked:
                del self.tracked_faces[tid]
        # Output
        smoothed = []
        for face in self.tracked_faces.values():
            det = ((float(face['score']), tuple(map(int, face['box']))), face['landmarks'])
            if face['emotion'] is not None:
                det = det + (face['emotion'],)
            smoothed.append(det)
        return smoothed

    def __preprocess_images(self, imgs, width, height):
        result = []
        for img in imgs:
            # Resize the image and change dims to fit neural network input.
            input_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]
            result.append(input_img)
        return np.array(result)

    def __detect_faces(self, img):
        fd_input, fd_output = self.face_detection_model.input(0), self.face_detection_model.output(0)
        fd_height, fd_width = list(fd_input.shape)[2:4]

        input_img = self.__preprocess_images([img], fd_width, fd_height)[0]
        results = self.face_detection_model([input_img])[fd_output]
        return self.__process_detection_results(img, results, fd_width, fd_height, thresh=0.25)

    def __detect_landmarks(self, img, boxes):
        fl_input, fl_output = self.face_landmarks_model.input(0), self.face_landmarks_model.output(0)
        fl_height, fl_width = list(fl_input.shape)[2:4]

        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = self.__preprocess_images(patches, fl_width, fl_height)
        # there are many faces on the image
        results = [self.face_landmarks_model([patch])[fl_output].squeeze() for patch in patches]
        return self.__process_landmark_results(boxes, results)

    def __recognize_emotions(self, img, boxes):
        fe_input, fe_output = self.emotions_recognition_model.input(0), self.emotions_recognition_model.output(0)
        fe_height, fe_width = list(fe_input.shape)[2:4]

        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = self.__preprocess_images(patches, fe_width, fe_height)
        # there are many faces on the image
        results = [self.emotions_recognition_model([patch])[fe_output].squeeze() for patch in patches]

        if not results:
            return []

        # map result to labels
        labels = list(map(lambda x: self.emotion_classes[x], np.argmax(results, axis=1)))
        return labels

    def __process_detection_results(self, frame, results, in_width, in_height, thresh=0.5):
        # The size of the original frame.
        h, w = frame.shape[:2]
        scale_x, scale_y = w, h
        # The 'results' variable is a [200, 5] tensor.
        results = results.squeeze()
        boxes = []
        scores = []
        for _, _, score, xmin, ymin, xmax, ymax in results:
            # Create a box with pixels real coordinates from the output box.
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            boxes.append(tuple(map(int, (xmin * scale_x, ymin * scale_y, (xmax - xmin) * scale_x, (ymax - ymin) * scale_y))))
            scores.append(float(score))

        # Apply non-maximum suppression to get rid of many overlapping entities.
        # See https://paperswithcode.com/method/non-maximum-suppression
        # This algorithm returns indices of objects to keep.
        indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)

        # If there are no boxes.
        if len(indices) == 0:
            return []

        # Filter detected objects.
        return [(scores[idx], boxes[idx]) for idx in indices.flatten()]

    def __process_landmark_results(self, boxes, results):
        landmarks = []

        for box, result in zip(boxes, results):
            # create a vector of landmarks (35x2)
            result = result.reshape(-1, 2)
            box = box[1]
            # move every landmark according to box origin
            landmarks.append((result * box[2:] + box[:2]).astype(np.int32))

        return landmarks

    def _draw_mask(self, img, mask_img, center, face_size, scale=1.0, offset_coeffs=(0.5, 0.5)):
        face_width, face_height = face_size

        # scale mask to fit face size
        mask_width = max(1.0, face_width * scale)
        f_scale = mask_width / mask_img.shape[1]
        mask_img = cv2.resize(mask_img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_AREA)

        x_offset_coeff, y_offset_coeff = offset_coeffs

        # left-top and right-bottom points
        x1, y1 = center[0] - int(mask_img.shape[1] * x_offset_coeff), center[1] - int(
            mask_img.shape[0] * y_offset_coeff)
        x2, y2 = x1 + mask_img.shape[1], y1 + mask_img.shape[0]

        # if points inside image
        if 0 < x2 < img.shape[1] and 0 < y2 < img.shape[0] or 0 < x1 < img.shape[1] and 0 < y1 < img.shape[1]:
            # face image to be overlayed
            face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
            # overlay
            mask_img = mask_img[max(0, -y1):max(0, -y1) + face_crop.shape[0],
                       max(0, -x1):max(0, -x1) + face_crop.shape[1]]
            
            # alpha channel to blend images
            alpha_mask = mask_img[:, :, 3:4] / 255.0
            alpha_bg = 1.0 - alpha_mask

            # blend images
            face_crop[:] = (alpha_mask * mask_img[:, :, :3] + alpha_bg * face_crop).astype(np.uint8)

    def __draw_santa(self, img, detection):
        (score, box), landmarks, emotion = detection
        # draw beard
        self._draw_mask(img, self.assets["santa_beard"], landmarks[5], box[2:], offset_coeffs=(0.5, 0))
        # draw cap
        self._draw_mask(img, self.assets["santa_cap"], np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.5,
                        offset_coeffs=(0.56, 0.78))

    def __draw_reindeer(self, img, landmarks, box):
        # draw antlers
        self._draw_mask(img, self.assets["reindeer_antlers"], np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.8,
                        offset_coeffs=(0.5, 1.1))
        # draw sunglasses
        self._draw_mask(img, self.assets["reindeer_sunglasses"], np.mean(landmarks[:4], axis=0, dtype=np.int32), box[2:],
                        offset_coeffs=(0.5, 0.33))
        # draw nose
        self._draw_mask(img, self.assets["reindeer_nose"], landmarks[4], box[2:], scale=0.25)


class HalloweenTheme(Theme):
    def __init__(self, device: str = "CPU"):
        super().__init__()
        self.default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (20, 21), (23, 24), (26, 27), (29, 30))

        self.assets = self._load_assets(["pumpkin"])

        self.model_precision = "FP16"
        self.device = device
        self.point_score_threshold = 0.25

        self.decoder = OpenPoseDecoder()
        self.pose_estimation_model = None

        self.tracked_poses = {}
        self.next_pose_id = 0
        self.smoothing_tau = 0.1
        self.matching_threshold = 80 # pixels, for centroid matching

        self.load_models(device)

    def load_models(self, device: str):
        self.pose_estimation_model = self._load_model("human-pose-estimation-0005", self.model_precision, device)

    def run_inference(self, frame: np.ndarray) -> Any:
        pe_input, pe_output_heatmaps, pe_output_embeddings = self.pose_estimation_model.input(0), self.pose_estimation_model.output("heatmaps"), self.pose_estimation_model.output("embeddings")
        height, width = list(pe_input.shape)[2:4]

        input_img = self.__preprocess_image(frame, width, height)

        results = self.pose_estimation_model([input_img])
        embeddings = results[pe_output_embeddings]
        heatmaps = results[pe_output_heatmaps]

        # Get poses from network results.
        poses, scores = self.__process_results(frame, embeddings, heatmaps)
        poses, scores = self._smooth_detections(poses, scores)
        # add additional points to skeletons
        poses = [self.__add_artificial_points(pose, self.point_score_threshold) for pose in poses]
        return list(zip(poses, scores))

    def _smooth_detections(self, poses, scores):
        now = time.time()

        if len(poses) == 0:
            self.tracked_poses = {}
            return [], []

        curr_centroids = [np.mean([pt[:2] for pt in pose if pt[2] > self.point_score_threshold], axis=0) if np.any(pose[:,2] > self.point_score_threshold) else np.zeros(2) for pose in poses]

        # Prepare tracked centroids
        tracked_ids = list(self.tracked_poses.keys())
        tracked_centroids = [self.tracked_poses[tid]['centroid'] for tid in tracked_ids]

        # Matching (using L2 distance between centroids)
        matches = []
        matched_tracked = set()
        matched_current = set()
        if tracked_ids:
            dists = np.linalg.norm(np.expand_dims(tracked_centroids,1)-np.expand_dims(curr_centroids,0), axis=2)
            for t_idx, tid in enumerate(tracked_ids):
                c_idx = np.argmin(dists[t_idx])
                if dists[t_idx, c_idx] < self.matching_threshold and c_idx not in matched_current:
                    matches.append((tid, c_idx))
                    matched_tracked.add(tid)
                    matched_current.add(c_idx)

        # Update matched tracks
        for tid, c_idx in matches:
            prev = self.tracked_poses[tid]
            dt = now - prev['last_update']
            alpha = np.exp(-dt / self.smoothing_tau)
            smoothed_pose = prev['pose'] * alpha + poses[c_idx] * (1 - alpha)
            smoothed_score = prev['score'] * alpha + scores[c_idx] * (1 - alpha)
            centroid = curr_centroids[c_idx]
            self.tracked_poses[tid].update({'pose': smoothed_pose, 'score': smoothed_score, 'centroid': centroid, 'last_update': now})

        # Add new tracks
        for c_idx, centroid in enumerate(curr_centroids):
            if c_idx not in matched_current:
                tid = self.next_pose_id
                self.next_pose_id += 1
                self.tracked_poses[tid] = {
                    'pose': poses[c_idx].copy(),
                    'score': scores[c_idx],
                    'centroid': centroid,
                    'last_update': now
                }

        # Remove old tracks
        for tid in tracked_ids:
            if tid not in matched_tracked:
                del self.tracked_poses[tid]

        # Output
        smoothed_poses = [track['pose'] for track in self.tracked_poses.values()]
        smoothed_scores = [track['score'] for track in self.tracked_poses.values()]
        return smoothed_poses, smoothed_scores

    def draw_results(self, image: np.ndarray, poses: Any) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.multiply(img, 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if len(poses) == 0:
            return img

        for pose, score in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]

            out_thickness = img.shape[0] // 100
            if points_scores[5] > self.point_score_threshold and points_scores[6] > self.point_score_threshold:
                out_thickness = max(2, abs(points[5, 0] - points[6, 0]) // 15)
            in_thickness = out_thickness // 2

            img_limbs = np.copy(img)
            # Draw limbs.
            for i, j in self.default_skeleton:
                if i < len(points_scores) and j < len(points_scores) and points_scores[i] > self.point_score_threshold and \
                        points_scores[j] > self.point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=(0, 0, 0), thickness=out_thickness,
                             lineType=cv2.LINE_AA)
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=(255, 255, 255),
                             thickness=in_thickness, lineType=cv2.LINE_AA)
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > self.point_score_threshold:
                    cv2.circle(img_limbs, tuple(p), 1, color=(0, 0, 0), thickness=2 * out_thickness,
                               lineType=cv2.LINE_AA)
                    cv2.circle(img_limbs, tuple(p), 1, color=(255, 255, 255), thickness=2 * in_thickness,
                               lineType=cv2.LINE_AA)

            cv2.addWeighted(img, 0.3, img_limbs, 0.7, 0, dst=img)

            face_size_scale = 2.2
            left_ear = 3
            right_ear = 4
            left_eye = 1
            right_eye = 2
            # if left eye and right eye and left ear or right ear are visible
            if points_scores[left_eye] > self.point_score_threshold and points_scores[
                right_eye] > self.point_score_threshold and (
                    points_scores[left_ear] > self.point_score_threshold or points_scores[
                right_ear] > self.point_score_threshold):
                # visible left ear and right ear
                if points_scores[left_ear] > self.point_score_threshold and points_scores[right_ear] > self.point_score_threshold:
                    face_width = np.linalg.norm(points[left_ear] - points[right_ear]) * face_size_scale
                    face_center = (points[left_ear] + points[right_ear]) // 2
                # visible left ear and right eye
                elif points_scores[left_ear] > self.point_score_threshold and points_scores[
                    right_eye] > self.point_score_threshold:
                    face_width = np.linalg.norm(points[left_ear] - points[right_eye]) * face_size_scale
                    face_center = (points[left_ear] + points[right_eye]) // 2
                # visible right ear and left eye
                elif points_scores[left_eye] > self.point_score_threshold and points_scores[
                    right_ear] > self.point_score_threshold:
                    face_width = np.linalg.norm(points[left_eye] - points[right_ear]) * face_size_scale
                    face_center = (points[left_eye] + points[right_ear]) // 2

                face_width = max(1.0, face_width)
                scale = face_width / self.assets["pumpkin"].shape[1]
                pumpkin_face = cv2.resize(self.assets["pumpkin"], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # left-top and right-bottom points
                x1, y1 = face_center[0] - pumpkin_face.shape[1] // 2, face_center[1] - pumpkin_face.shape[0] * 2 // 3
                x2, y2 = face_center[0] + pumpkin_face.shape[1] // 2, face_center[1] + pumpkin_face.shape[0] // 3

                # face image to be overlayed
                face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
                # overlay
                pumpkin_face = pumpkin_face[max(0, -y1):max(0, -y1) + face_crop.shape[0],
                               max(0, -x1):max(0, -x1) + face_crop.shape[1]]
                # alpha channel to blend images
                alpha_pumpkin = pumpkin_face[:, :, 3:4] / 255.0
                alpha_bg = 1.0 - alpha_pumpkin

                # blend images
                face_crop[:] = (alpha_pumpkin * pumpkin_face)[:, :, :3] + alpha_bg * face_crop

        return img

    def __add_artificial_points(self, pose, point_score_threshold):
        # neck, bellybutton, ribs
        neck = (pose[5] + pose[6]) / 2
        bellybutton = (pose[11] + pose[12]) / 2
        if neck[2] > point_score_threshold and bellybutton[2] > point_score_threshold:
            rib_1_center = (neck + bellybutton) / 2
            rib_1_left = (pose[5] + bellybutton) / 2
            rib_1_right = (pose[6] + bellybutton) / 2
            rib_2_center = (neck + rib_1_center) / 2
            rib_2_left = (pose[5] + rib_1_left) / 2
            rib_2_right = (pose[6] + rib_1_right) / 2
            rib_3_center = (neck + rib_2_center) / 2
            rib_3_left = (pose[5] + rib_2_left) / 2
            rib_3_right = (pose[6] + rib_2_right) / 2
            rib_4_center = (rib_1_center + rib_2_center) / 2
            rib_4_left = (rib_1_left + rib_2_left) / 2
            rib_4_right = (rib_1_right + rib_2_right) / 2
            new_points = [neck, bellybutton, rib_1_center, rib_1_left, rib_1_right, rib_2_center, rib_2_left, rib_2_right,
                          rib_3_center, rib_3_left, rib_3_right, rib_4_center, rib_4_left, rib_4_right]
            pose = np.vstack([pose, new_points])
        return pose

    def __preprocess_image(self, img, width, height):
        input_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]
        return input_img

    def __process_results(self, img, embeddings, heatmaps):
        def heatmap_nms(heatmaps, pooled_heatmaps):
            return heatmaps * (heatmaps == pooled_heatmaps)

        # 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
        def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
            # Padding
            A = np.pad(A, padding, mode="constant")

            # Window view of A
            output_shape = (
                (A.shape[0] - kernel_size) // stride + 1,
                (A.shape[1] - kernel_size) // stride + 1,
            )
            kernel_size = (kernel_size, kernel_size)
            A_w = as_strided(
                A,
                shape=output_shape + kernel_size,
                strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
            )
            A_w = A_w.reshape(-1, *kernel_size)

            # Return the result of pooling.
            if pool_mode == "max":
                return A_w.max(axis=(1, 2)).reshape(output_shape)
            elif pool_mode == "avg":
                return A_w.mean(axis=(1, 2)).reshape(output_shape)

        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
        )
        nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

        # Decode poses.
        poses, scores = self.decoder(heatmaps, nms_heatmaps, embeddings)
        output_shape = list(self.pose_estimation_model.output(index=0).partial_shape)
        output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores


class EasterTheme(ChristmasTheme):
    def __init__(self, device: str = "CPU"):
        super().__init__(device)
        self.assets = self._load_assets(["bunny_ears", "bunny_boss_ears", "bunny_nose", "bunny_tie"])

    def draw_results(self, image: np.ndarray, detections: Any) -> np.ndarray:
        if not detections:
            return image

        # sort by face size
        detections = list(sorted(detections, key=lambda x: x[0][1][2] * x[0][1][3]))

        for (score, box), landmarks, emotion in detections[:-1]:
            self.__draw_bunny(image, landmarks, box)

        # the largest face is bunny boss
        self.__draw_bunny_boss(image, detections[-1][1], detections[-1][0][1])

        return image

    def __draw_bunny_boss(self, image, landmarks, box):
        # draw ears
        self._draw_mask(image, self.assets["bunny_boss_ears"], np.mean(landmarks[13:17], axis=0, dtype=np.int32),
                        box[2:], scale=1.8, offset_coeffs=(0.5, 1.25))
        # draw nose
        self._draw_mask(image, self.assets["bunny_nose"], landmarks[4], box[2:], scale=0.85, offset_coeffs=(0.5, 0.2))

    def __draw_bunny(self, image, landmarks, box):
        # draw ears
        self._draw_mask(image, self.assets["bunny_ears"], np.mean(landmarks[13:17], axis=0, dtype=np.int32),
                        box[2:], scale=1.3, offset_coeffs=(0.5, 1.25))

        # draw tie
        self._draw_mask(image, self.assets["bunny_tie"], landmarks[26], box[2:], scale=0.6, offset_coeffs=(0.5, 0.0))


class WildTheme(ChristmasTheme):
    def __init__(self, device: str = "CPU"):
        super().__init__(device)
        # Override assets with bear and raccoon faces
        self.assets = self._load_assets(["bear", "raccoon"])

    def draw_results(self, image: np.ndarray, detections: Any) -> np.ndarray:
        if not detections:
            return image

        for (score, box), landmarks, emotion in detections:
            # Calculate face size
            face_size = max(box[2], box[3])  # width or height, whichever is larger
            
            # Draw bear for faces larger than 1/8 of frame width, raccoon for smaller faces
            if face_size > image.shape[1] / 8:
                self.__draw_bear(image, landmarks, box)
            else:
                self.__draw_raccoon(image, landmarks, box)

        return image

    def __draw_bear(self, image, landmarks, box):
        # Draw bear face using landmarks for positioning
        self._draw_mask(image, self.assets["bear"], 
                       np.mean(landmarks[:4], axis=0, dtype=np.int32),  # Use eye landmarks for positioning
                       box[2:], scale=2.2, offset_coeffs=(0.5, 0.5))

    def __draw_raccoon(self, image, landmarks, box):
        # Draw raccoon face using landmarks for positioning
        self._draw_mask(image, self.assets["raccoon"], 
                       np.mean(landmarks[:4], axis=0, dtype=np.int32),  # Use eye landmarks for positioning
                       box[2:], scale=2.2, offset_coeffs=(0.5, 0.5))
