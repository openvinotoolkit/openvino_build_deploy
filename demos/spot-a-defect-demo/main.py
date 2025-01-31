import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from anomalib.data import MVTec, NumpyImageBatch
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.metrics import F1Max, Evaluator
from anomalib.models import get_model
from lightning.pytorch.callbacks import EarlyStopping
from supervision import BoxCornerAnnotator, LabelAnnotator, TraceAnnotator, LineZoneAnnotator, LineZone, ByteTrack, \
    Point, Detections, Color, ColorLookup, \
    DetectionsSmoother
from supervision.annotators.base import BaseAnnotator
from torchvision.transforms import v2
from ultralytics import YOLOWorld, YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor

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


def load_yolo_model(model_name: str) -> YOLOWorld:
    ov_model_path = MODEL_DIR / f"{model_name}_openvino_model"

    if not ov_model_path.exists():
        model = YOLO(MODEL_DIR / f"{model_name}.pt")
        model.export(format="openvino", dynamic=False, half=True)

    model = YOLOWorld(model_name)
    # set classes to detect
    model.set_classes([MAIN_CLASS] + AUX_CLASSES + NULL_CLASSES)

    config = {'batch': 1, 'conf': 0.01, 'imgsz': 640, 'mode': 'predict', 'model': ov_model_path, 'save': False}
    predictor = DetectionPredictor(overrides=config)
    predictor.setup_model(model=ov_model_path, verbose=False)
    # todo: set predictor to model
    # model.predictor = predictor

    return model


def train_anomalib_model(model_name: str) -> Path:
    # augmentation are needed for mitigating domain shift
    augmentations = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally with 50% probability
        v2.RandomVerticalFlip(p=0.2),  # Randomly flip images vertically with 20% probability
        v2.RandomRotation(degrees=30),  # Randomly rotate images within a range of ±30 degrees
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop and resize images
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Randomly adjust colors
        v2.RandomGrayscale(p=0.1),  # Convert images to grayscale with 10% probability
    ])

    val_metrics = [F1Max(fields=["pred_score", "gt_label"])]
    evaluator = Evaluator(val_metrics=val_metrics, test_metrics=val_metrics)
    early_stopping = EarlyStopping(monitor="F1Max", patience=3, mode="max")

    # define the class to train on
    datamodule = MVTec(DATA_DIR / "mvtec", category="hazelnut", augmentations=augmentations)
    model = get_model(model_name)
    model.evaluator = evaluator

    # validation and testing are not needed for this demo
    engine = Engine(max_epochs=10, limit_test_batches=0, callbacks=[early_stopping], accelerator="cpu")
    engine.fit(datamodule=datamodule, model=model)
    # export model to openvino
    return engine.export(model, ExportType.OPENVINO, MODEL_DIR / model_name, input_size=(256, 256))


def load_anomalib_model(model_name: str) -> OpenVINOInferencer:
    model_path = MODEL_DIR / model_name / "weights" / "openvino" / "model.xml"
    if not model_path.exists():
        model_path = train_anomalib_model(model_name)

    return OpenVINOInferencer(model_path)


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


def draw_results(frame: np.ndarray, annotators: list[BaseAnnotator], line_zone: LineZone, detections: Detections, anomaly_results: NumpyImageBatch, visualize: str) -> None:
    for annotator in annotators:
        if isinstance(annotator, LineZoneAnnotator):
            annotator.annotate(frame, line_counter=line_zone)
        else:
            annotator.annotate(frame, detections)

    # todo anomaly visualization
    # for box, anomaly in zip(detections.xyxy, anomaly_results):
    #     x1, y1, x2, y2 = add_box_margin(box, frame.shape[:2][::-1])
    #     anomaly_score = float(anomaly.pred_score)
    #
    #     # draw anomaly map
    #     if visualize == "heatmaps":
    #         anomaly_map = cv2.normalize(anomaly.anomaly_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #         colormap = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
    #         colormap = cv2.resize(colormap, (x2 - x1, y2 - y1))
    #         frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.6, colormap, 0.4, 0)
    #     # draw anomaly mask
    #     elif visualize == "masks":
    #         pred_mask = anomaly.pred_mask.astype(np.uint8) * 255
    #         mask = np.zeros_like(pred_mask)
    #         mask = np.dstack((mask, mask, pred_mask))
    #         mask = cv2.resize(mask, (x2 - x1, y2 - y1))
    #         frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.6, mask, 0.4, 0)
    #
    #     color = (0, 0, 255) if anomaly_score > 0.5 else (0, 255, 0)
    #     # draw a red rectangle around the object
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    #     # draw anomaly score
    #     utils.draw_text(frame, text=f"Anomaly: {anomaly_score:.2f}", point=(int(x1), int(y1)))


def run(video_path: str, det_model_name: str, anomaly_model_name: str, visualize: str, flip: bool):
    det_model = load_yolo_model(det_model_name)
    anomaly_model = load_anomalib_model(anomaly_model_name)

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

        patches = get_patches(frame, det_results)
        anomalies = anomaly_model.predict(patches) if len(patches) > 0 else NumpyImageBatch(frame)

        end_time = time.time()

        draw_results(frame, annotators, line_zone, det_results, anomalies, visualize)

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
    parser.add_argument("--detection_model", type=str, default="yolov8s-worldv2", help="Model for object detection",
                        choices=["yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world",
                                 "yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"])
    parser.add_argument("--anomaly_model", type=str, default="Stfpm", help="Model for anomaly detection")
    parser.add_argument("--visualize", type=str, default=None, help="Visualization type",
                        choices=["heatmaps", "masks"])
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run(args.stream, args.detection_model, args.anomaly_model, args.visualize, args.flip)