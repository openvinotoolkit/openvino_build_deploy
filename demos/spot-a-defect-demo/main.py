import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from anomalib.data import MVTec
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import get_model
from ultralytics import YOLOWorld

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
DATA_DIR = Path("data")


def load_yolo_model(model_name: str) -> YOLOWorld:
    model = YOLOWorld(MODEL_DIR / f"{model_name}.pt")
    # set classes to detect
    model.set_classes(["hazelnut", "nut"])
    # todo convert model to OV
    # path = model.export(format="openvino", dynamic=False, half=True)
    # model =  YOLO(path, task="detect")
    return model


def train_anomalib_model(model_name: str):
    # define the class to train on
    datamodule = MVTec(DATA_DIR / "mvtec", category="hazelnut")
    model = get_model(model_name)

    engine = Engine(max_epochs=1)
    engine.fit(datamodule=datamodule, model=model)
    # export model to openvino
    engine.export(model, ExportType.OPENVINO, MODEL_DIR / model_name)


def load_anomalib_model(model_name: str) -> OpenVINOInferencer:
    model_path = MODEL_DIR / model_name / "weights" / "openvino" / "model.xml"
    if not model_path.exists():
        train_anomalib_model(model_name)

    return OpenVINOInferencer(model_path)


def run(video_path: str, det_model_name: str, anomaly_model_name: str, flip: bool):
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
        results = det_model.predict(frame, conf=0.01)[0]
        end_time = time.time()

        processing_times.append(end_time - start_time)
        # mean processing time [ms]
        processing_time = np.mean(processing_times) * 1000

        frame = results.plot()
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
    parser.add_argument("--anomaly_model", type=str, default="Padim", help="Model for anomaly detection")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run(args.stream, args.detection_model, args.anomaly_model, args.flip)