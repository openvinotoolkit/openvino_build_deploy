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

warnings.filterwarnings("ignore", category=UserWarning, module="cv2")  # Suppress PNG warnings

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils
from decoder import OpenPoseDecoder

# Pose estimation skeleton for spooky theme
DEFAULT_SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (20, 21), (23, 24), (26, 27), (29, 30))

# Emotion classes and mapping for Santa theme
EMOTION_CLASSES = ["neutral", "happy", "sad", "surprise", "anger"]
EMOTION_MAPPING = {"neutral": "Rudolph", "happy": "Cupid", "sad": "Prancer", "surprise": "Blitzen", "anger": "Vixen"}

# Load assets with existence checks
pumpkin_img = cv2.imread("assets/pumpkin.png", cv2.IMREAD_UNCHANGED) if os.path.exists("assets/pumpkin.png") else None
santa_beard_img = cv2.imread("assets/santa_beard.png", cv2.IMREAD_UNCHANGED) if os.path.exists(
    "assets/santa_beard.png") else None
santa_cap_img = cv2.imread("assets/santa_cap.png", cv2.IMREAD_UNCHANGED) if os.path.exists(
    "assets/santa_cap.png") else None
reindeer_nose_img = cv2.imread("assets/reindeer_nose.png", cv2.IMREAD_UNCHANGED) if os.path.exists(
    "assets/reindeer_nose.png") else None
reindeer_sunglasses_img = cv2.imread("assets/reindeer_sunglasses.png", cv2.IMREAD_UNCHANGED) if os.path.exists(
    "assets/reindeer_sunglasses.png") else None
reindeer_antlers_img = cv2.imread("assets/reindeer_antlers.png", cv2.IMREAD_UNCHANGED) if os.path.exists(
    "assets/reindeer_antlers.png") else None


def download_model(model_name, precision):
    base_model_dir = Path("model")
    model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"
    if not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(model_url_dir + model_name + '.bin', model_path.with_suffix('.bin').name, model_path.parent)
        utils.download_file(model_url_dir + model_name + '.xml', model_path.name, model_path.parent)
    return model_path


def load_model(model_path, device):
    core = ov.Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer


def preprocess_images(imgs, width, height):
    result = []
    for img in imgs:
        input_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...]
        result.append(input_img)
    return np.array(result)


def process_detection_results(frame, results, in_width, in_height, thresh=0.5):
    h, w = frame.shape[:2]
    scale_x, scale_y = w / in_width, h / in_height
    results = results.squeeze()
    boxes = []
    scores = []
    for xmin, ymin, xmax, ymax, score in results:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        boxes.append(
            tuple(map(int, (xmin * scale_x, ymin * scale_y, (xmax - xmin) * scale_x, (ymax - ymin) * scale_y))))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)
    return [(scores[idx], boxes[idx]) for idx in indices.flatten()] if len(indices) > 0 else []


def process_landmark_results(boxes, results):
    landmarks = []
    for box, result in zip(boxes, results):
        result = result.reshape(-1, 2)
        box = box[1]
        landmarks.append((result * box[2:] + box[:2]).astype(np.int32))
    return landmarks


def draw_mask(img, mask_img, center, face_size, scale=1.0, offset_coeffs=(0.5, 0.5)):
    if mask_img is None:
        return
    face_width, face_height = face_size
    mask_width = max(1.0, face_width * scale)
    f_scale = mask_width / mask_img.shape[1]
    mask_img = cv2.resize(mask_img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_AREA)
    x_offset_coeff, y_offset_coeff = offset_coeffs
    x1, y1 = center[0] - int(mask_img.shape[1] * x_offset_coeff), center[1] - int(mask_img.shape[0] * y_offset_coeff)
    x2, y2 = x1 + mask_img.shape[1], y1 + mask_img.shape[0]
    if 0 < x2 < img.shape[1] and 0 < y2 < img.shape[0] or 0 < x1 < img.shape[1] and 0 < y1 < img.shape[1]:
        face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
        mask_img = mask_img[max(0, -y1):max(0, -y1) + face_crop.shape[0], max(0, -x1):max(0, -x1) + face_crop.shape[1]]
        alpha_pumpkin = mask_img[:, :, 3:4] / 255.0
        alpha_bg = 1.0 - alpha_pumpkin
        face_crop[:] = (alpha_pumpkin * mask_img)[:, :, :3] + alpha_bg * face_crop


# Santa-specific drawing functions
def draw_santa(img, detection):
    (score, box), landmarks, emotion = detection
    draw_mask(img, santa_beard_img, landmarks[5], box[2:], offset_coeffs=(0.5, 0))
    draw_mask(img, santa_cap_img, np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.5,
              offset_coeffs=(0.56, 0.78))


def draw_reindeer(img, landmarks, box):
    draw_mask(img, reindeer_antlers_img, np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.8,
              offset_coeffs=(0.5, 1.1))
    draw_mask(img, reindeer_sunglasses_img, np.mean(landmarks[:4], axis=0, dtype=np.int32), box[2:],
              offset_coeffs=(0.5, 0.33))
    draw_mask(img, reindeer_nose_img, landmarks[4], box[2:], scale=0.25)


def draw_christmas_masks(frame, detections):
    detections = list(sorted(detections, key=lambda x: x[0][1][2] * x[0][1][3]))
    if not detections:
        return frame
    for (score, box), landmarks, emotion in detections[:-1]:
        draw_reindeer(frame, landmarks, box)
        (label_width, label_height), _ = cv2.getTextSize(
            text=EMOTION_MAPPING[emotion],
            fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            fontScale=box[2] / 150,
            thickness=1)
        point = np.mean(landmarks[:4], axis=0, dtype=np.int32) - [label_width // 2, 2 * label_height]
        cv2.putText(
            img=frame,
            text=EMOTION_MAPPING[emotion],
            org=point,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            fontScale=box[2] / 150,
            color=(0, 0, 196),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    draw_santa(frame, detections[-1])
    return frame


# Spooky-specific functions
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    A = np.pad(A, padding, mode="constant")
    output_shape = ((A.shape[0] - kernel_size) // stride + 1, (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.max(axis=(1, 2)).reshape(output_shape) if pool_mode == "max" else A_w.mean(axis=(1, 2)).reshape(
        output_shape)


def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


def process_poses(img, pafs, heatmaps, model, decoder):
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    poses[:, :, :2] *= output_scale
    return poses, scores


def add_artificial_points(poses, scores, skeleton=DEFAULT_SKELETON, threshold=0.25):
    if poses.size == 0:
        return poses, scores

    for i in range(len(poses)):
        pose = poses[i]
        keypoint_scores = pose[:, 2]
        valid_points = pose[keypoint_scores > threshold, :2]
        valid_indices = np.where(keypoint_scores > threshold)[0]

        if len(valid_points) < 2:
            continue

        new_pose = pose.copy()

        for j, (start, end) in enumerate(skeleton):
            if start in valid_indices and end not in valid_indices:
                direction = np.mean(valid_points - valid_points[0], axis=0)
                new_pose[end, :2] = new_pose[start, :2] + direction * 0.5
                new_pose[end, 2] = threshold / 2
            elif end in valid_indices and start not in valid_indices:
                direction = np.mean(valid_points - valid_points[0], axis=0)
                new_pose[start, :2] = new_pose[end, :2] - direction * 0.5
                new_pose[start, 2] = threshold / 2

        poses[i] = new_pose

    return poses, scores


def draw_spooky(img, poses, faces, landmarks):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.multiply(img, 0.5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        keypoint_scores = pose[:, 2]
        for i, j in DEFAULT_SKELETON:
            if i < len(keypoint_scores) and j < len(keypoint_scores) and keypoint_scores[i] > 0.25 and keypoint_scores[j] > 0.25:
                cv2.line(img, tuple(points[i]), tuple(points[j]), (0, 0, 0), 2, cv2.LINE_AA)
                cv2.line(img, tuple(points[i]), tuple(points[j]), (255, 255, 255), 1, cv2.LINE_AA)
    for (_, box), lm in zip(faces, landmarks):
        face_center = np.mean(lm[:4], axis=0, dtype=np.int32)
        draw_mask(img, pumpkin_img, face_center, box[2:], scale=2.2)
    return img


def run_demo(source, theme, face_detection_model, face_landmarks_model, face_emotions_model, pose_model_name,
             model_precision, device, flip):
    device_mapping = utils.available_devices()
    decoder = OpenPoseDecoder()

    # Load models
    face_detection_model_path = download_model(face_detection_model, model_precision)
    face_landmarks_model_path = download_model(face_landmarks_model, model_precision)
    face_emotions_model_path = download_model(face_emotions_model, model_precision)
    pose_model_path = download_model(pose_model_name, model_precision)

    fd_model, fd_input, fd_output = load_model(face_detection_model_path, device)
    fl_model, fl_input, fl_output = load_model(face_landmarks_model_path, device)
    fe_model, fe_input, fe_output = load_model(face_emotions_model_path, device)
    pose_model = ov.Core().compile_model(ov.Core().read_model(pose_model_path), device)

    fd_height, fd_width = list(fd_input.shape)[2:4]
    fl_height, fl_width = list(fl_input.shape)[2:4]
    fe_height, fe_width = list(fe_input.shape)[2:4]
    pose_input_shape = (256, 456)  # Adjusted for pose model

    def detect_faces(img):
        input_img = preprocess_images([img], fd_width, fd_height)[0]
        results = fd_model([input_img])[fd_output]
        return process_detection_results(img, results, fd_width, fd_height, thresh=0.25)

    def detect_landmarks(img, boxes):
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fl_width, fl_height)
        results = [fl_model([patch])[fl_output].squeeze() for patch in patches]
        return process_landmark_results(boxes, results)

    def recognize_emotions(img, boxes):
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fe_width, fe_height)
        results = [fe_model([patch])[fe_output].squeeze() for patch in patches]
        return [EMOTION_CLASSES[np.argmax(result)] for result in results] if results else []

    player = None
    try:
        source = int(source) if isinstance(source, str) and source.isnumeric() else source
        player = utils.VideoPlayer(source=source, flip=flip, size=(1280, 720), fps=30)  # Reduced resolution
        player.start()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        
        # Set window position and size
        cv2.moveWindow(title, 0, 0)  # Move window to position (100, 100)
        cv2.resizeWindow(title, 1280, 720)  # Resize window to 1280x720

        processing_times = collections.deque(maxlen=200)  # Limit the size of the deque
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            start_time = time.time()

            boxes = detect_faces(frame)
            landmarks = detect_landmarks(frame, boxes)
            emotions = recognize_emotions(frame, boxes) if theme == "santa" else []

            if theme == "spooky":
                pose_input = cv2.resize(frame, pose_input_shape[::-1], interpolation=cv2.INTER_AREA).transpose((2, 0, 1))[np.newaxis, ...]
                pose_results = pose_model([pose_input])
                poses, scores = process_poses(frame, pose_results[pose_model.output("Mconv7_stage2_L1")],
                                              pose_results[pose_model.output("Mconv7_stage2_L2")], pose_model, decoder)
                poses, scores = add_artificial_points(poses, scores)  # Add artificial points for spooky theme
                frame = draw_spooky(frame, poses, boxes, landmarks)
                effect_text = f"Poses: {len(poses)}, Pumpkins: {len(boxes)}"
            elif theme == "santa":
                detections = zip(boxes, landmarks, emotions)
                frame = draw_christmas_masks(frame, detections)
                effect_text = f"Faces: {len(boxes)}, Santa: 1, Reindeer: {max(0, len(boxes) - 1)}" if boxes else "No faces detected"

            stop_time = time.time()
            processing_times.append(stop_time - start_time)

            utils.draw_ov_watermark(frame)
            fps = 1000 / (np.mean(processing_times) * 1000)
            utils.draw_text(frame, f"Theme: {theme.capitalize()}", (10, 10))
            utils.draw_text(frame, effect_text, (10, 30))
            utils.draw_text(frame, f"FPS: {fps:.1f}", (10, 50))

            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

            for i, dev in enumerate(device_mapping.keys()):
                if key == ord('1') + i:
                    del fd_model, fl_model, fe_model, pose_model
                    fd_model, fd_input, fd_output = load_model(face_detection_model_path, dev)
                    fl_model, fl_input, fl_output = load_model(face_landmarks_model_path, dev)
                    fe_model, fe_input, fe_output = load_model(face_emotions_model_path, dev)
                    pose_model = ov.Core().compile_model(ov.Core().read_model(pose_model_path), dev)
                    device = dev
                    processing_times.clear()

    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            player.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument('--theme', default="spooky", choices=["spooky", "santa"], help="Theme to apply")
    parser.add_argument('--device', default="AUTO", type=str, help="Device to start inference on")
    parser.add_argument("--detection_model_name", type=str, default="face-detection-0205", help="Face detection model")
    parser.add_argument("--landmarks_model_name", type=str, default="facial-landmarks-35-adas-0002",
                        help="Face landmarks model")
    parser.add_argument("--emotions_model_name", type=str, default="emotions-recognition-retail-0003",
                        help="Face emotions model")
    parser.add_argument("--pose_model_name", type=str, default="human-pose-estimation-0001",
                        help="Pose estimation model")
    parser.add_argument("--model_precision", type=str, default="FP16-INT8", choices=["FP16-INT8", "FP16", "FP32"],
                        help="Model precision")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_demo(args.stream, args.theme, args.detection_model_name, args.landmarks_model_name,
             args.emotions_model_name, args.pose_model_name, args.model_precision, args.device, args.flip)