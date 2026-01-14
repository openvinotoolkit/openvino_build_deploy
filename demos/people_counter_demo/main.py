import argparse
import json
import logging as log
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
import supervision as sv
from supervision import ColorLookup
from ultralytics import YOLO

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils
from deep_sort_realtime.deepsort_tracker import DeepSort

CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def convert(model_name: str, model_dir: Path) -> tuple[Path, Path]:
    model_path = model_dir / f"{model_name}.pt"
    # create a YOLO object detection model
    yolo_model = YOLO(model_path)

    ov_model_path = model_dir / f"{model_name}_openvino_model"
    ov_int8_model_path = model_dir / f"{model_name}_int8_openvino_model"
    # export the model to OpenVINO format (FP16 and INT8)
    if not ov_model_path.exists():
        ov_model_path = yolo_model.export(format="openvino", dynamic=False, half=True)
    if not ov_int8_model_path.exists():
        ov_int8_model_path = yolo_model.export(format="openvino", dynamic=False, half=True, int8=True, data="coco128.yaml")
    return Path(ov_model_path), Path(ov_int8_model_path)


def get_model(model_path: Path, verbose: bool = False):
    # compile the model with YOLO
    model = YOLO(model_path, task="detect", verbose=verbose)
    return model


def load_zones(json_path: str) -> List[np.ndarray]:
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)

    # return a list of zones defined by points
    return [np.array(zone["points"], np.int32) for zone in zones_dict.values()]


def get_annotators(json_path: str, colorful: bool = False) -> Tuple[List, List, List, List, List]:
    # list of points
    polygons = load_zones(json_path)

    # colors for zones
    colors = sv.ColorPalette.DEFAULT

    zones = []
    zone_annotators = []
    box_annotators = []
    masks_annotators = []
    label_annotators = []
    for index, polygon in enumerate(polygons, start=1):
        # a zone to count objects in
        zone = sv.PolygonZone(polygon=polygon)
        zones.append(zone)
        # the annotator - visual part of the zone
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=0))
        # box annotator, showing boxes around objects
        box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.BoxAnnotator(color=colors.by_idx(index))
        box_annotators.append(box_annotator)
        # mask annotator, showing transparent mask
        mask_annotator = sv.MaskAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.MaskAnnotator(color=colors.by_idx(index))
        masks_annotators.append(mask_annotator)
        # label annotator, showing people ids
        label_annotator = sv.LabelAnnotator(text_scale=0.7, color_lookup=ColorLookup.INDEX) if colorful else sv.LabelAnnotator(text_scale=0.7, color=colors.by_idx(index))
        label_annotators.append(label_annotator)

    return zones, zone_annotators, box_annotators, masks_annotators, label_annotators


def track_objects(frame: np.array, detections: sv.Detections, tracker: DeepSort) -> List:
    # Convert detections to the format required by the tracker
    detection_list = []
    for det in zip(detections.xyxy, detections.confidence):
        bbox = det[0].tolist()
        confidence = det[1]
        detection_list.append((bbox, confidence))
 
    # Update the tracker with the new detections
    tracks = tracker.update_tracks(detection_list, frame=frame)
    # detections are sorted by confidence so tracks must be also sorted the same way
    return list(sorted(tracks, key=lambda x: x.det_conf if x.det_conf is not None else 0.0, reverse=True))


def draw_annotations(frame: np.array, detections: sv.Detections, tracker: DeepSort, queue_count: Dict, object_limit: int, category:str,
                     zones: List, zone_annotators: List, box_annotators: List, masks_annotators: List, label_annotators: List) -> None:

    for zone_annotator in zone_annotators:
        # visualize polygon for the zone
        frame = zone_annotator.annotate(scene=frame)

    if detections:
        # uniquely track the objects
        tracks = track_objects(frame, detections, tracker)

        # annotate the frame with the detected persons within each zone
        for zone_id, (zone, box_annotator, masks_annotator, label_annotator) in enumerate(
                zip(zones, box_annotators, masks_annotators, label_annotators), start=1):

            # get detections relevant only for the zone
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # visualize boxes around objects in the zone
            frame = masks_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            # count how many objects detected
            det_count = len(detections_filtered)
            # Add track ID annotations
            label_annotator.annotate(scene=frame, detections=detections_filtered,
                                     labels=[f"ID: {track.track_id if track.time_since_update == 0 else ' '}" for
                                             track, in_zone in zip(tracks, mask) if in_zone])
            # add the count to the list
            queue_count[zone_id].append(det_count)
            # calculate the mean number of customers in the queue
            mean_customer_count = np.mean(queue_count[zone_id], dtype=np.int32)

            cat = "people" if category == "person" else "objects"
            # add alert text to the frame if necessary, flash every second
            if mean_customer_count > object_limit and time.time() % 2 > 1:
                utils.draw_text(frame, text=f"Too many {cat} in zone {zone_id}!", point=(frame.shape[1] // 2, frame.shape[0] // 2), center=True, font_color=(0, 0, 255))

            # print an info about number of customers in the queue, ask for the more assistants if required
            log.info(
                f"Zone {zone_id}, avg {category} count: {mean_customer_count} {f'Too many {cat}!' if mean_customer_count > object_limit else ''}")


def run(video_path: str, model_paths: Tuple[Path, Path], model_name: str = "", category: str = "person", zones_config_file: str = "",
        object_limit: int = 3, flip: bool = True, tracker_frames: int = 1800, colorful: bool = False, last_frames: int = 50) -> None:

    model_mapping = {
        "FP16": model_paths[0],
        "INT8": model_paths[1],
    }

    # Start with INT8
    model_type = "INT8"
    model = get_model(model_mapping[model_type], verbose=False)

    # Device setup
    devices_mapping = utils.available_devices()  # e.g. {"cpu":"Intel CPU", "gpu":"Intel GPU", ...}
    device_type = next(iter(devices_mapping.keys()))  # default to first available

    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/people_counter_demo", with_embedded_image=True)

    # Video player
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)

    # get zones, and zone and box annotators for zones
    zones, zone_annotators, box_annotators, masks_annotators, label_annotators = get_annotators(json_path=zones_config_file, colorful=colorful)
    category_id = CATEGORIES.index(category)

    # object counter
    queue_count = defaultdict(lambda: deque(maxlen=last_frames))
    # keep at most 100 last times
    processing_times = deque(maxlen=100)

    # Initialize the tracker with a higher max_age
    tracker = DeepSort(max_age=tracker_frames, n_init=3)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # start a video stream
    player.start()
    while True:
        frame = player.next()
        if frame is None:
            print("Source ended")
            break
        # Get the results.
        frame = np.array(frame)
        f_height, f_width = frame.shape[:2]

        # inference + timing
        result = model(frame, device=f"intel:{device_type}", verbose=False)[0]
        processing_times.append(result.speed["inference"])

        # convert to supervision detections
        detections = sv.Detections.from_ultralytics(result)
        # filter out other predictions than selected category
        detections = detections[detections.class_id == category_id]

        # draw results
        draw_annotations(frame, detections, tracker, queue_count, object_limit, category, zones, zone_annotators, box_annotators, masks_annotators, label_annotators)

        # Mean processing time [ms].
        processing_time = np.mean(processing_times)

        fps = 1000 / processing_time
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(10, 10))
        utils.draw_text(frame, text=f"Currently running {model_name} ({model_type}) on {device_type}", point=(10, 50))

        # Draw control panel & watermark
        utils.draw_control_panel(frame, devices_mapping)
        utils.draw_ov_watermark(frame)
        utils.draw_qr_code(frame, qr_code)

        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key in (27, ord('q')):
            break

        # handle keypress for precision/device
        model_changed = False

        if key == ord('f'):
            model_type = "FP16"
            model_changed = True
        elif key == ord('i'):
            model_type = "INT8"
            model_changed = True
        for i, dev in enumerate(devices_mapping.keys()):
            if key == ord('1') + i:
                device_type = dev
                model_changed = True

        if model_changed:
            del model
            model = get_model(model_mapping[model_type], verbose=False)
            processing_times.clear()

    player.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Model version to be converted",
                        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo26l-seg", "yolo26x-seg",
                                 "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
                                 "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"])
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--category', type=str, default="person", choices=CATEGORIES, help="The category to detect (from COCO dataset)")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--object_limit', type=int, default=3, help="The maximum number of objects in the area")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument('--colorful', action="store_true", help="If objects should be annotated with random colors")
    parser.add_argument('--tracker_frames', type=int, default=1800, help="Maximum number of missed frames for the tracker")

    args = parser.parse_args()
    model_paths = convert(args.model_name, Path(args.model_dir))
    run(args.stream, model_paths, args.model_name, args.category, args.zones_config_file, args.object_limit, args.tracker_frames, args.flip, args.colorful)
