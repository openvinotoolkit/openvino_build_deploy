import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from supervision import BoxCornerAnnotator, LabelAnnotator, TraceAnnotator, LineZoneAnnotator, LineZone, ByteTrack, \
    Point, Detections, Color, ColorLookup, DetectionsSmoother
from supervision.annotators.base import BaseAnnotator
from ultralytics import YOLOWorld
from ultralytics.engine.results import Results

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
DATA_DIR = Path("data")

MAIN_CLASS = "hazelnut"
AUX_CLASSES = ["nut", "brown ball"]
# the following are "null" classes to improve detection of the main classes
NULL_CLASSES = ["person", "hand", "finger", "fabric"]

PROBABILITY_COEFFICIENT = 15


def load_yolo_model(model_name: str, device: str) -> YOLOWorld:
    model = YOLOWorld(model_name)
    opts = {"device": device, "config": {"PERFORMANCE_HINT": "LATENCY"}, "model_caching" : True, "cache_dir": "cache"}
    model = torch.compile(model, backend="openvino", options=opts)
    # set classes to detect
    model.set_classes([MAIN_CLASS] + AUX_CLASSES + NULL_CLASSES)

    return model


def load_annotators(size: tuple[int, int]) -> tuple[list[BaseAnnotator], LineZone, ByteTrack, DetectionsSmoother]:
    line_x = int(0.3 * size[0])
    start = Point(line_x, 0)
    end = Point(line_x, size[1])

    line_zone = LineZone(start=start, end=end)
    box_annotator = BoxCornerAnnotator(corner_length=int(0.03 * size[0]), color_lookup=ColorLookup.TRACK)
    label_annotator = LabelAnnotator(text_scale=0.7, color_lookup=ColorLookup.TRACK)
    trace_annotator = TraceAnnotator(thickness=box_annotator.thickness, color_lookup=ColorLookup.TRACK)
    line_zone_annotator = LineZoneAnnotator(thickness=box_annotator.thickness, color=Color.RED, text_scale=label_annotator.text_scale,
                                            display_in_count=False, custom_out_text="Count")

    tracker = ByteTrack()
    smoother = DetectionsSmoother(length=3)

    return [box_annotator, label_annotator, trace_annotator, line_zone_annotator], line_zone, tracker, smoother


def add_box_margin(box: tuple[int], frame_size: tuple[int], margin_ratio: float = 0.15) -> tuple[int]:
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    border_x = width * margin_ratio
    border_y = height * margin_ratio

    new_x1 = max(0, x1 - border_x)
    new_y1 = max(0, y1 - border_y)
    new_x2 = min(frame_w, x2 + border_x)
    new_y2 = min(frame_h, y2 + border_y)

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def filter_and_process_results(det_results: Results, tracker: ByteTrack, smoother: DetectionsSmoother, line_zone: LineZone) -> Detections:
    detections = Detections.from_ultralytics(det_results)
    # we have to increase probabilities, which are very low in case of YOLOWorld to be able to use tracking
    detections.confidence *= PROBABILITY_COEFFICIENT

    # filter out the null classes
    detections = detections[np.isin(detections.class_id, np.arange(len(AUX_CLASSES) + 1))]
    # set the class_id to 0 (MAIN_CLASS) for all detections
    detections.data["class_name"] = [MAIN_CLASS] * len(detections)

    detections = detections.with_nmm(class_agnostic=True)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    line_zone.trigger(detections)

    return detections


def get_patches(frame: np.ndarray, results: Detections) -> np.ndarray:
    patches = []

    for box, _, _, _, _, _ in results:
        # add border to the bounding box to fit the training data
        x1, y1, x2, y2 = add_box_margin(box, frame.shape[:2][::-1])
        patch = frame[y1:y2, x1:x2]
        patch = cv2.resize(patch, (256, 256))
        patches.append(patch)

    return np.array(patches)


def draw_results(frame: np.ndarray, annotators: list[BaseAnnotator], line_zone: LineZone, detections: Detections) -> None:
    for annotator in annotators:
        if isinstance(annotator, LineZoneAnnotator):
            annotator.annotate(frame, line_counter=line_zone)
        else:
            annotator.annotate(frame, detections)


def run(video_path: str, det_model_name: str, device: str, flip: bool):
    det_model = load_yolo_model(det_model_name, device)

    video_size = (1920, 1080)
    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=video_size, fps=60, flip=flip)
    annotators, line_zone, tracker, smoother = load_annotators(video_size)

    processing_times = deque(maxlen=100)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # start a video stream
    player.start()
    while True:
        # Grab the frame.
        frame = player.next()
        if frame is None:
            print("Source ended")
            break

        f_height, f_width = frame.shape[:2]

        start_time = time.time()

        det_results = det_model.predict(frame, conf=0.01, verbose=False)[0]
        det_results = filter_and_process_results(det_results, tracker, smoother, line_zone)

        end_time = time.time()

        draw_results(frame, annotators, line_zone, det_results)

        processing_times.append(end_time - start_time)
        # mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000

        fps = 1000 / processing_time
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(f_width * 7 // 10, 10))

        utils.draw_ov_watermark(frame)
        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

    player.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument('--device', default="AUTO", type=str, help="Device to run inference on")
    parser.add_argument("--detection_model", type=str, default="yolov8s-worldv2", help="Model for object detection",
                        choices=["yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world",
                                 "yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"])
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run(args.stream, args.detection_model, args.device, args.flip)
