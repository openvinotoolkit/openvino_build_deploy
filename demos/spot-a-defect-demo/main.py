import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from anomalib.data import MVTec, NumpyImageBatch
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.metrics import F1Max, Evaluator
from anomalib.models import get_model
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.transforms import v2
from ultralytics import YOLOWorld, YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
DATA_DIR = Path("data")

MAIN_CLASSES = ["hazelnut", "nut"]
# the following are "null" classes to improve detection of the main classes
NULL_CLASSES = ["person", "hand", "finger", "fabric"]


def load_yolo_model(model_name: str) -> YOLOWorld:
    ov_model_path = MODEL_DIR / f"{model_name}_openvino_model"

    if not ov_model_path.exists():
        model = YOLO(MODEL_DIR / f"{model_name}.pt")
        model.export(format="openvino", dynamic=False, half=True)

    model = YOLOWorld(model_name)
    # set classes to detect
    model.set_classes(MAIN_CLASSES + NULL_CLASSES)

    config = {'batch': 1, 'conf': 0.01, 'imgsz': 640, 'mode': 'predict', 'model': ov_model_path, 'save': False}
    predictor = DetectionPredictor(overrides=config)
    predictor.setup_model(model=ov_model_path, verbose=False)
    # todo: set predictor to model
    # model.predictor = predictor

    return model


def train_anomalib_model(model_name: str):
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
    engine.export(model, ExportType.OPENVINO, MODEL_DIR / model_name, input_size=(256, 256))


def load_anomalib_model(model_name: str) -> OpenVINOInferencer:
    model_path = MODEL_DIR / model_name / "weights" / "openvino" / "model.xml"
    if not model_path.exists():
        train_anomalib_model(model_name)

    return OpenVINOInferencer(model_path)


def add_box_margin(row, frame_size, margin_ratio=0.15):
    w, h = frame_size
    width = row["x2"] - row["x1"]
    height = row["y2"] - row["y1"]

    border_x = width * margin_ratio
    border_y = height * margin_ratio

    new_x1 = max(0, row["x1"] - border_x)
    new_y1 = max(0, row["y1"] - border_y)
    new_x2 = min(w, row["x2"] + border_x)
    new_y2 = min(h, row["y2"] + border_y)

    return pd.Series([new_x1, new_y1, new_x2, new_y2], index=["x1", "y1", "x2", "y2"]).astype(int)


def filter_and_convert_results(det_results: Results) -> pd.DataFrame:
    df_results = det_results.to_df(decimals=0)
    df_results[["x1", "y1", "x2", "y2"]] = np.nan

    if not df_results.empty:
        # split box column into 4 columns
        df_results[["x1", "y1", "x2", "y2"]] = pd.DataFrame(df_results["box"].tolist(), index=df_results.index)
        df_results[["x1", "y1", "x2", "y2"]] = df_results[["x1", "y1", "x2", "y2"]].astype(int)
        df_results.drop(columns=["box"], inplace=True)
        # filter out the null classes
        df_results = df_results[df_results["name"].isin(MAIN_CLASSES)]

    return df_results


def get_patches(frame: np.ndarray, results: pd.DataFrame) -> np.ndarray:
    patches = []

    for result in results.itertuples():
        x1, y1, x2, y2 = result.x1, result.y1, result.x2, result.y2
        patch = frame[int(y1):int(y2), int(x1):int(x2)]
        patch = cv2.resize(patch, (256, 256))
        patches.append(patch)

    return np.array(patches)


def draw_results(frame: np.ndarray, det_results: pd.DataFrame, anomaly_results: NumpyImageBatch, visualize: str) -> None:
    for box, anomaly in zip(det_results[["x1", "y1", "x2", "y2"]].to_numpy(), anomaly_results):
        x1, y1, x2, y2 = box
        anomaly_score = float(anomaly.pred_score)

        # draw anomaly map
        if visualize == "heatmaps":
            anomaly_map = cv2.normalize(anomaly.anomaly_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colormap = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
            colormap = cv2.resize(colormap, (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.6, colormap, 0.4, 0)
        # draw anomaly mask
        elif visualize == "masks":
            pred_mask = anomaly.pred_mask.astype(np.uint8) * 255
            mask = np.zeros_like(pred_mask)
            mask = np.dstack((mask, mask, pred_mask))
            mask = cv2.resize(mask, (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.6, mask, 0.4, 0)

        color = (0, 0, 255) if anomaly_score > 0.5 else (0, 255, 0)
        # draw a red rectangle around the object
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # draw anomaly score
        utils.draw_text(frame, text=f"Anomaly: {anomaly_score:.2f}", point=(int(x1), int(y1)))


def run(video_path: str, det_model_name: str, anomaly_model_name: str, visualize: str, flip: bool):
    det_model = load_yolo_model(det_model_name)
    anomaly_model = load_anomalib_model(anomaly_model_name)

    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)

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

        det_results = filter_and_convert_results(det_results)
        # add border to the bounding box to fit the training data
        det_results = det_results.apply(lambda x: add_box_margin(x, (f_width, f_height)), axis=1)

        patches = get_patches(frame, det_results)
        anomalies = anomaly_model.predict(patches) if len(patches) > 0 else NumpyImageBatch(frame)

        end_time = time.time()

        draw_results(frame, det_results, anomalies, visualize)

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