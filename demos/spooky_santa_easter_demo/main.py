from utils import demo_utils as utils
from decoder import OpenPoseDecoder
import argparse
import collections
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from numpy.lib.stride_tricks import as_strided

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="cv2")  # Suppress PNG warnings


SCRIPT_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "..",
    "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))


# Pose estimation skeleton
DEFAULT_SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (20, 21), (23, 24), (26, 27), (29, 30))

# Emotion mapping for Santa theme
EMOTION_CLASSES = ["neutral", "happy", "sad", "surprise", "anger"]
EMOTION_MAPPING = {
    "neutral": "Rudolph",
    "happy": "Cupid",
    "sad": "Prancer",
    "surprise": "Blitzen",
    "anger": "Vixen"}

# Load assets with existence checks
pumpkin_img = cv2.imread(
    "assets/pumpkin.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/pumpkin.png") else None
santa_beard_img = cv2.imread(
    "assets/santa_beard.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/santa_beard.png") else None
santa_cap_img = cv2.imread(
    "assets/santa_cap.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/santa_cap.png") else None
reindeer_nose_img = cv2.imread(
    "assets/reindeer_nose.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/reindeer_nose.png") else None
reindeer_sunglasses_img = cv2.imread(
    "assets/reindeer_sunglasses.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/reindeer_sunglasses.png") else None
reindeer_antlers_img = cv2.imread(
    "assets/reindeer_antlers.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/reindeer_antlers.png") else None
bunny_ears_img = cv2.imread(
    "assets/bunny_ears.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/bunny_ears.png") else None
egg_img = cv2.imread(
    "assets/egg.png",
    cv2.IMREAD_UNCHANGED) if os.path.exists("assets/egg.png") else None

# Pooling and NMS for pose estimation


def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    A = np.pad(A, padding, mode="constant")
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape +
        kernel_size,
        strides=(
            stride *
            A.strides[0],
            stride *
            A.strides[1]) +
        A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.max(
        axis=(
            1,
            2)).reshape(output_shape) if pool_mode == "max" else A_w.mean(
        axis=(
            1,
            2)).reshape(output_shape)


def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


def process_poses(img, pafs, heatmaps, model, decoder):
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / \
        output_shape[2].get_length()
    poses[:, :, :2] *= output_scale
    return poses, scores


def process_faces(frame, results, in_width, in_height, thresh=0.5):
    h, w = frame.shape[:2]
    scale_x, scale_y = w / in_width, h / in_height
    results = results.squeeze()
    boxes, scores = [], []
    for xmin, ymin, xmax, ymax, score in results:
        xmin, ymin, xmax, ymax = max(
            0, xmin), max(
            0, ymin), min(
            w, xmax), min(
                h, ymax)
        boxes.append(tuple(map(int, (xmin * scale_x, ymin * scale_y,
                     (xmax - xmin) * scale_x, (ymax - ymin) * scale_y))))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=thresh,
        nms_threshold=0.6)
    return [(scores[idx], boxes[idx])
            for idx in indices.flatten()] if len(indices) > 0 else []


def process_landmarks(boxes, results):
    return [(result.reshape(-1, 2) * box[2:] + box[:2]).astype(np.int32)
            for (_, box), result in zip(boxes, results)]


def process_emotions(results):
    return [EMOTION_CLASSES[np.argmax(result)]
            for result in results] if results else []


def draw_mask(
    img,
    mask_img,
    center,
    size,
    scale=1.0,
    offset_coeffs=(
        0.5,
        0.5)):
    if mask_img is None:
        return
    width = max(1, int(size[0] * scale))
    f_scale = width / mask_img.shape[1]
    mask_img = cv2.resize(
        mask_img,
        None,
        fx=f_scale,
        fy=f_scale,
        interpolation=cv2.INTER_AREA)
    x1, y1 = center[0] - int(mask_img.shape[1] * offset_coeffs[0]
                             ), center[1] - int(mask_img.shape[0] * offset_coeffs[1])
    x2, y2 = x1 + mask_img.shape[1], y1 + mask_img.shape[0]
    if 0 < x2 < img.shape[1] and 0 < y2 < img.shape[0] or 0 < x1 < img.shape[1] and 0 < y1 < img.shape[1]:
        crop = img[max(0, y1):min(y2, img.shape[0]),
                   max(0, x1):min(x2, img.shape[1])]
        mask_crop = mask_img[max(
            0, -y1):max(0, -y1) + crop.shape[0], max(0, -x1):max(0, -x1) + crop.shape[1]]
        alpha = mask_crop[:, :, 3:4] / 255.0
        crop[:] = (alpha * mask_crop[:, :, :3] +
                   (1 - alpha) * crop).astype(np.uint8)

# New function added from the second code


def add_artificial_points(pose, point_score_threshold):
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
        new_points = [
            neck,
            bellybutton,
            rib_1_center,
            rib_1_left,
            rib_1_right,
            rib_2_center,
            rib_2_left,
            rib_2_right,
            rib_3_center,
            rib_3_left,
            rib_3_right,
            rib_4_center,
            rib_4_left,
            rib_4_right]
        pose = np.vstack([pose, new_points])
    return pose

# Enhanced draw_spooky function incorporating logic from draw_poses


def draw_spooky(img, poses, faces, landmarks):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.multiply(img, 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if poses.size == 0 and not faces:
        return img

    point_score_threshold = 0.25
    for pose in poses:
        pose = add_artificial_points(pose, point_score_threshold)
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]

        out_thickness = img.shape[0] // 100
        if points_scores[5] > point_score_threshold and points_scores[6] > point_score_threshold:
            out_thickness = max(2, abs(points[5, 0] - points[6, 0]) // 15)
        in_thickness = out_thickness // 2

        img_limbs = np.copy(img)
        # Draw limbs
        for i, j in DEFAULT_SKELETON:
            if i < len(points_scores) and j < len(
                    points_scores) and points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(
                    img_limbs, tuple(
                        points[i]), tuple(
                        points[j]), (0, 0, 0), out_thickness, cv2.LINE_AA)
                cv2.line(
                    img_limbs, tuple(
                        points[i]), tuple(
                        points[j]), (255, 255, 255), in_thickness, cv2.LINE_AA)
        # Draw joints
        for p, v in zip(points, points_scores):
            if v > point_score_threshold:
                cv2.circle(img_limbs, tuple(p), 1, (0, 0, 0),
                           2 * out_thickness, cv2.LINE_AA)
                cv2.circle(img_limbs, tuple(p), 1, (255, 255, 255),
                           2 * in_thickness, cv2.LINE_AA)
        cv2.addWeighted(img, 0.3, img_limbs, 0.7, 0, dst=img)

    # Use face landmarks if available, otherwise use pose points for pumpkin
    # overlay
    for (_, box), lm in zip(faces, landmarks):
        face_center = np.mean(lm[:4], axis=0, dtype=np.int32)
        draw_mask(img, pumpkin_img, face_center, box[2:], scale=2.2)
    if not faces:  # Fallback to pose-based face detection
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            face_size_scale = 2.2
            left_ear, right_ear, left_eye, right_eye = 3, 4, 1, 2
            if (points_scores[left_eye] > point_score_threshold and points_scores[right_eye] > point_score_threshold and (
                    points_scores[left_ear] > point_score_threshold or points_scores[right_ear] > point_score_threshold)):
                if points_scores[left_ear] > point_score_threshold and points_scores[right_ear] > point_score_threshold:
                    face_width = np.linalg.norm(
                        points[left_ear] - points[right_ear]) * face_size_scale
                    face_center = (points[left_ear] + points[right_ear]) // 2
                elif points_scores[left_ear] > point_score_threshold and points_scores[right_eye] > point_score_threshold:
                    face_width = np.linalg.norm(
                        points[left_ear] - points[right_eye]) * face_size_scale
                    face_center = (points[left_ear] + points[right_eye]) // 2
                elif points_scores[left_eye] > point_score_threshold and points_scores[right_ear] > point_score_threshold:
                    face_width = np.linalg.norm(
                        points[left_eye] - points[right_ear]) * face_size_scale
                    face_center = (points[left_eye] + points[right_ear]) // 2

                face_width = max(1.0, float(face_width))
                scale = face_width / pumpkin_img.shape[1]
                pumpkin_face = cv2.resize(
                    pumpkin_img,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA)
                x1, y1 = face_center[0] - pumpkin_face.shape[1] // 2, face_center[1] - \
                    pumpkin_face.shape[0] * 2 // 3
                x2, y2 = face_center[0] + pumpkin_face.shape[1] // 2, face_center[1] + \
                    pumpkin_face.shape[0] // 3
                face_crop = img[max(0, y1):min(y2, img.shape[0]), max(
                    0, x1):min(x2, img.shape[1])]
                pumpkin_face = pumpkin_face[max(
                    0, -y1):max(0, -y1) + face_crop.shape[0], max(0, -x1):max(0, -x1) + face_crop.shape[1]]
                alpha_pumpkin = pumpkin_face[:, :, 3:4] / 255.0
                face_crop[:] = (alpha_pumpkin *
                                pumpkin_face[:, :, :3] +
                                (1 -
                                 alpha_pumpkin) *
                                face_crop).astype(np.uint8)

    return img


def draw_santa(img, faces, landmarks, emotions):
    if not faces:
        return img
    detections = sorted(zip(faces, landmarks, emotions),
                        key=lambda x: x[0][1][2] * x[0][1][3])
    for (score, box), lm, emo in detections[:-1]:
        draw_mask(img, reindeer_antlers_img, np.mean(
            lm[13:17], axis=0, dtype=np.int32), box[2:], scale=1.8, offset_coeffs=(0.5, 1.1))
        draw_mask(img, reindeer_nose_img, lm[4], box[2:], scale=0.25)
        cv2.putText(img, EMOTION_MAPPING[emo], tuple(
            lm[4] - [20, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 196), 1, cv2.LINE_AA)
    (_, box), lm, _ = detections[-1]
    draw_mask(img, santa_beard_img, lm[5], box[2:], offset_coeffs=(0.5, 0))
    draw_mask(img,
              santa_cap_img,
              np.mean(lm[13:17],
                      axis=0,
                      dtype=np.int32),
              box[2:],
              scale=1.5,
              offset_coeffs=(0.56,
                             0.78))
    return img


def draw_easter(img, poses, faces, landmarks):
    for pose in poses:
        torso_center = np.mean(
            pose[5:7], axis=0, dtype=np.int32) if pose[5][2] > 0.25 and pose[6][2] > 0.25 else None
        if torso_center is not None:
            draw_mask(img, egg_img, torso_center, (100, 150), scale=1.8)
    for (_, box), lm in zip(faces, landmarks):
        draw_mask(img, bunny_ears_img, np.mean(
            lm[13:17], axis=0, dtype=np.int32), box[2:], scale=1.5, offset_coeffs=(0.5, 1.0))
    return img


def load_model(model_name, precision, device):
    base_model_dir = Path("model")
    model_path = base_model_dir / "intel" / \
        model_name / precision / f"{model_name}.xml"
    if not model_path.exists():
        model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(
            model_url + model_name + '.xml',
            model_path.name,
            model_path.parent)
        utils.download_file(
            model_url + model_name + '.bin',
            model_path.with_suffix('.bin').name,
            model_path.parent)
    core = ov.Core()
    model = core.read_model(model_path)
    return core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})


def run_demo(source, theme, model_precision, device, flip):
    decoder = OpenPoseDecoder()
    pose_model = load_model(
        "human-pose-estimation-0001",
        model_precision,
        device)
    face_model = load_model("face-detection-0205", model_precision, device)
    landmark_model = load_model(
        "facial-landmarks-35-adas-0002",
        model_precision,
        device)
    emotion_model = load_model(
        "emotions-recognition-retail-0003",
        model_precision,
        device)

    # Corrected: width, height for cv2.resize to match [1, 3, 256, 456]
    pose_input_shape = (456, 256)
    face_input_shape = list(face_model.input(0).shape)[2:4]
    landmark_input_shape = list(landmark_model.input(0).shape)[2:4]
    emotion_input_shape = list(emotion_model.input(0).shape)[2:4]

    source = int(source) if source.isnumeric() else source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(
            f"Warning: Cannot open webcam {source}. Falling back to default VideoPlayer behavior.")
    player = utils.VideoPlayer(source, flip=flip, fps=30, size=(1920, 1080))
    player.start()

    cv2.namedWindow("Themed Demo", cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(
        "Themed Demo",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)

    processing_times = collections.deque()
    while True:
        frame = player.next()
        if frame is None:
            print("No frame captured, skipping iteration")
            continue

        start_time = time.time()

        # Pose estimation with shape verification
        pose_input = cv2.resize(
            frame, pose_input_shape, interpolation=cv2.INTER_AREA).transpose(
            (2, 0, 1))[
            np.newaxis, ...]
        pose_results = pose_model([pose_input])
        poses, _ = process_poses(frame, pose_results[pose_model.output(
            "Mconv7_stage2_L1")], pose_results[pose_model.output("Mconv7_stage2_L2")], pose_model, decoder)

        # Face detection and processing
        face_input = cv2.resize(
            frame, face_input_shape, interpolation=cv2.INTER_AREA).transpose(
            (2, 0, 1))[
            np.newaxis, ...]
        faces = process_faces(
            frame,
            face_model(
                [face_input])[
                face_model.output(0)],
            face_input_shape[1],
            face_input_shape[0])
        patches = [frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                   for _, box in faces]
        landmarks = process_landmarks(
            faces, [
                landmark_model(
                    [
                        cv2.resize(
                            p, tuple(landmark_input_shape), interpolation=cv2.INTER_AREA).transpose(
                            (2, 0, 1))[
                            np.newaxis, ...]])[
                                landmark_model.output(0)].squeeze() for p in patches])
        emotions = process_emotions(
            [
                emotion_model(
                    [
                        cv2.resize(
                            p, tuple(emotion_input_shape), interpolation=cv2.INTER_AREA).transpose(
                            (2, 0, 1))[
                            np.newaxis, ...]])[
                                emotion_model.output(0)].squeeze() for p in patches])

        # Apply theme
        if theme == "spooky":
            frame = draw_spooky(frame, poses, faces, landmarks)
            effect_text = f"Poses: {len(poses)}, Pumpkins: {len(faces)}"
        elif theme == "santa":
            frame = draw_santa(frame, faces, landmarks, emotions)
            effect_text = f"Faces: {
                len(faces)}, Santa: 1, Reindeer: {
                max(
                    0, len(faces) - 1)}"
        elif theme == "easter":
            frame = draw_easter(frame, poses, faces, landmarks)
            effect_text = f"Poses: {len(poses)}, Bunnies: {len(faces)}"

        stop_time = time.time()
        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        utils.draw_ov_watermark(frame)
        fps = 1000 / (np.mean(processing_times) * 1000)
        utils.draw_text(frame, f"Theme: {theme.capitalize()}", (10, 10))
        utils.draw_text(frame, f"FPS: {fps:.1f}", (10, 50))

        cv2.imshow("Themed Demo", frame)
        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

    player.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="0", type=str,
                        help="Path to video file or webcam number")
    parser.add_argument("--theme", default="spooky",
                        choices=["spooky", "santa", "easter"], help="Theme to apply")
    parser.add_argument("--model_precision", default="FP16-INT8",
                        choices=["FP16-INT8", "FP16", "FP32"], help="Model precision")
    parser.add_argument("--device", default="AUTO", type=str,
                        help="Device for inference")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    args = parser.parse_args()
    run_demo(
        args.stream,
        args.theme,
        args.model_precision,
        args.device,
        args.flip)
