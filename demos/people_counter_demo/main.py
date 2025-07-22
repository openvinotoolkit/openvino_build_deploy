
import argparse
from collections import deque
from typing import List
import sys, os
import cv2
import json
import numpy as np
import logging
import time
from ultralytics import YOLO


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import demo_utils as utils

# Suppress Ultralytics verbose logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

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

def load_zones(json_path: str) -> List[np.ndarray]:
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)

    # return a list of zones defined by points
    return [np.array(zone["points"], np.int32) for zone in zones_dict.values()]


def run(video_path: str, model_name: str = "", category: str = "person", zones_config_file: str = "",
        object_limit: int = 3, flip: bool = True, tracker_frames: int = 1800, colorful: bool = False, last_frames: int = 50) -> None:
    # Export both FP16 & INT8 ONCE at startup
    pt_path  = f"{model_name}.pt"
    ir_fp16   = f"{model_name}_openvino_model"
    ir_int8   = f"{model_name}_int8_openvino_model"

    base = YOLO(pt_path)
    # Export FP16 if missing
    if not os.path.isdir(ir_fp16):
        logging.info(f"Exporting FP16 IR to {ir_fp16}...")
        base.export(format="openvino", half=True)
    else:
        logging.info(f"FP16 IR already exists at {ir_fp16}, skipping export.")
    # Export INT8 if missing
    if not os.path.isdir(ir_int8):
        logging.info(f"Exporting INT8 IR to {ir_int8}...")
        base.export(format="openvino", int8=True)
    else:
        logging.info(f"INT8 IR already exists at {ir_int8}, skipping export.")

    # Start with FP16
    current_precision = "FP16"
    model = YOLO(ir_fp16, task="detect", verbose=False)

    # Device setup
    devices_mapping = utils.available_devices()  # e.g. {"cpu":"Intel CPU", "gpu":"Intel GPU", ...}
    device_type = f"intel:{next(iter(devices_mapping.keys()))}"  # default to first available

    # Zones & history
    zones      = load_zones(zones_config_file)
    rects      = [cv2.boundingRect(z) for z in zones]
    history    = {i+1: deque(maxlen=last_frames) for i in range(len(zones))}
    last_log   = {i+1: 0 for i in range(len(zones))}
    times      = deque(maxlen=100)
    cls_target = CATEGORIES.index(category)  # should be 0 for "person"

    # Video player
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920,1080), fps=60, flip=flip)

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

        # inference + timing
        start_time  = time.time()
        res = model(frame, device=device_type, verbose=False)[0]
        dt  = (time.time() - start_time) * 1000  # ms
        times.append(dt)

        # start from raw frame
        annotated = frame.copy()

        # Draw each zone’s bounding box in red
        for zx, zy, zw, zh in rects:
            cv2.rectangle(annotated, (zx, zy), (zx + zw, zy + zh), (0, 0, 255), 2)

        # Extract boxes & classes
        boxes   = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        now     = time.time()

        # Gather all person detections
        person_indices = [(pid, box.astype(int))
            for pid, (box, cls_id) in enumerate(zip(boxes, classes), start=1)
            if cls_id == cls_target]

        # Draw each person’s bounding box in green and their ID
        for pid, (x1, y1, x2, y2) in person_indices:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            px, py = int((x1 + x2) / 2), y1 - 10
            utils.draw_text(annotated, text=f"ID:{pid}", point=(px, py),font_color=(0, 255, 0))

        # Draw total persons count
        total_p = len(person_indices)
        utils.draw_text(annotated, text=f"Total persons: {total_p}", point=(10, 10), font_color=(255,255,255))

         # Per-zone average
        for zid, ((zx, zy, zw, zh), hist) in enumerate(zip(rects, history.values()), start=1):
            # Count current persons in zone
            count = sum(1 for _, (x1, y1, x2, y2) in person_indices
                if not (x2 < zx or x1 > zx+zw or y2 < zy or y1 > zy+zh))
            # Record count history
            if count or hist:
                hist.append(count)
            # Calculate the mean number of customers in the queue
            mean_customer_count = int(np.mean(hist, dtype=np.int32)) if hist else 0

            # Log and alert
            if now - last_log[zid] >= 1:
                alert_msg = ' Intel employee required!' if mean_customer_count > object_limit else ''
                logging.info(f"Zone {zid}, avg {category} count: {mean_customer_count}{alert_msg}")
                last_log[zid] = now
            # Flash alert text every other second
            if mean_customer_count > object_limit and now % 2 > 1:
                utils.draw_text(annotated, text=f"Intel employee required in zone {zid}!", point=(20, 20), 
                                font_color=(0, 0, 255))
            # Draw zone average count
            utils.draw_text(annotated, text=f"Zone {zid}: {mean_customer_count}", point=(zx + 5, zy + 20),
                font_color=(0, 255, 255))

        # Performance metrics
        mean_t = sum(times)/len(times) if times else 0
        fps    = 1000/mean_t if mean_t>0 else 0
        h,w,_  = annotated.shape
        utils.draw_text(annotated, text=f"Inference: {mean_t:.0f}ms ({fps:.1f} FPS)", point=(w//2, 10))
        utils.draw_text(annotated, text=f"Model: {model_name} ({current_precision}) on {device_type}", point=(w//2, 50))

        # Draw control panel & watermark
        utils.draw_control_panel(annotated, devices_mapping)
        utils.draw_ov_watermark(annotated)

        cv2.imshow(title, annotated)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

        # handle keypress for precision/device
        model_changed = False

        if key == ord('f') and current_precision != "FP16":
            current_precision = "FP16"
            logging.info("Switched to FP16")
            model_changed = True
        elif key == ord('i') and current_precision != "INT8":
            current_precision = "INT8"
            logging.info("Switched to INT8")
            model_changed = True

        for idx, dev in enumerate(devices_mapping.keys()):
            if key == ord(str(idx+1)):
                device_type   = f"intel:{dev}"
                logging.info(f"Switched to device {device_type}")
                model_changed = True

        if model_changed:
            del model
            folder = ir_fp16 if current_precision=="FP16" else ir_int8
            model = YOLO(folder, task="detect", verbose=False)
            times.clear()

    player.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, default="yolo11n", help="Model version to be converted",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
                                 "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"])
    parser.add_argument('--category', type=str, default="person", choices=CATEGORIES, help="The category to detect (from COCO dataset)")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--object_limit', type=int, default=3, help="The maximum number of objects in the area")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument('--colorful', action="store_true", help="If objects should be annotated with random colors")
    parser.add_argument('--tracker_frames', type=int, default=1800, help="Maximum number of missed frames for the tracker")

    args = parser.parse_args()
    run(args.stream, args.model_name, args.category, args.zones_config_file, args.object_limit, args.tracker_frames, args.flip, args.colorful)
    
