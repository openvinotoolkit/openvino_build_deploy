import argparse
import collections
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from ultralytics.engine.results import Results

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


def export_model(model_name: str) -> Path:
    model_dir = Path("model")
    model_path = model_dir / f"{model_name}.pt"
    # create a YOLO pose estimation model
    yolo_model = YOLO(model_path)

    ov_model_path = model_dir / f"{model_name}_int8_openvino_model"
    # export the model to OpenVINO format (INT8)
    if not ov_model_path.exists():
        yolo_model.export(format="openvino", dynamic=False, int8=True)
    return ov_model_path / f"{model_name}.xml"


def load_and_compile_model(model_path: Path, device: str) -> YOLO:
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model=model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY"})

    pose_model = YOLO(model_path.parent, task="pose")

    if pose_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        pose_model.predictor = pose_model._smart_load("predictor")(overrides={**pose_model.overrides, **custom}, _callbacks=pose_model.callbacks)
        pose_model.predictor.setup_model(model=pose_model.model)

    pose_model.predictor.model.ov_compiled_model = compiled_model

    return pose_model


colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def draw_poses(img: np.ndarray, detections: Results, point_score_threshold: float = 0.5, skeleton: Tuple[Tuple[int, int]] = default_skeleton):
    keypoints = detections.keypoints
    poses = keypoints.xy.numpy()
    scores = keypoints.conf.numpy() if keypoints.conf is not None else np.ones_like(poses[..., 0])
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose, score in zip(poses, scores):
        points = pose.astype(np.int32)
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, score)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if score[i] > point_score_threshold and score[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def run_pose_estimation(source: str, model_name: str, device: str, flip: bool = True) -> None:
    device_mapping = utils.available_devices()

    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/strike_a_pose_demo", with_embedded_image=True)

    model_path = export_model(model_name)
    pose_model = load_and_compile_model(model_path, device)

    player = None
    try:
        if isinstance(source, str) and source.isnumeric():
            source = int(source)
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=30, size=(1920, 1080))
        # Start capturing.
        player.start()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = pose_model(frame, verbose=False)[0]
            stop_time = time.time()

            # Draw poses on a frame.
            frame = draw_poses(frame, results)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            utils.draw_text(frame, text=f"Currently running {model_name} (INT8) on {device}", point=(10, 10))
            utils.draw_text(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (10, 50))

            # Draw watermark
            utils.draw_ov_watermark(frame)
            utils.draw_qr_code(frame, qr_code)

            cv2.imshow(title, frame)
            key = cv2.waitKey(1)

            # escape = 27 or 'q' to close the app
            if key == 27 or key == ord('q'):
                break

            for i, dev in enumerate(device_mapping.keys()):
                if key == ord('1') + i:
                    del pose_model
                    device = dev

                    pose_model = load_and_compile_model(model_path, device)

                    processing_times.clear()
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument('--device', default="AUTO", type=str, help="Device to run inference on")
    parser.add_argument("--model_name", default="yolo11n-pose",  type=str, help="Model version to be converted",
                        choices=["yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
                                 "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose"])
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_pose_estimation(args.stream, args.model_name, args.device, args.flip)
