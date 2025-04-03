from IPython import display

import spaces
import supervision as sv
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter
import logging as log
import json
import torch
import time
import uuid


import glob
import pandas as pd
import six
import datetime
import re

import gradio as gr
import webbrowser
from threading import Timer

log.basicConfig(level=log.ERROR)

# Support Functions


def load_zones(json_path, zone_str):
    """
        Load zones specified in an external json file
        Parameters:
            json_path: path to the json file with defined zones
            zone_str:  name of the zone in the json file
        Returns:
           zones: a list of arrays with zone points
    """
    # load json file
    with open(json_path) as f:
        zones_dict = json.load(f)
    # return a list of zones defined by points
    return np.array(zones_dict[zone_str]["points"], np.int32)


def draw_text(image, text, point, color=(255, 255, 255)) -> None:
    """
    Draws text

    Parameters:
        image: image to draw on
        text: text to draw
        point:
        color: text color
    """
    _, f_width = image.shape[:2]

    text_size, _ = cv2.getTextSize(
        text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2)

    rect_width = text_size[0] + 20
    rect_height = text_size[1] + 20
    rect_x, rect_y = point

    cv2.rectangle(image, pt1=(rect_x, rect_y), pt2=(rect_x + rect_width,
                  rect_y + rect_height), color=(255, 255, 255), thickness=cv2.FILLED)

    text_x = (rect_x + (rect_width - text_size[0]) // 2) - 10
    text_y = (rect_y + (rect_height + text_size[1]) // 2) - 10

    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=color, thickness=2, lineType=cv2.LINE_AA)


def get_iou(person_det, object_det):
    # Obtain the Intersection
    x_left = max(person_det[0], object_det[0])
    y_top = max(person_det[1], object_det[1])
    x_right = min(person_det[2], object_det[2])
    y_bottom = min(person_det[3], object_det[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    person_area = (person_det[2] - person_det[0]) * \
        (person_det[3] - person_det[1])
    obj_area = (object_det[2] - object_det[0]) * \
        (object_det[3] - object_det[1])

    return intersection_area / float(person_area + obj_area - intersection_area)


def intersecting_bboxes(bboxes, person_bbox, action_str):
    # Identify if person and object bounding boxes are intersecting using IOU
    for box in bboxes:
        if box.cls == 0:
            # If it is a person
            try:
                person_bbox.append([box.xyxy[0], box.id.numpy().astype(int)])
            except:
                pass
        elif box.cls != 0 and len(person_bbox) >= 1:
            # If it is not a person and an interaction took place with a person
            for p_bbox in person_bbox:
                if box.cls != 0:
                    result_iou = get_iou(p_bbox[0], box.xyxy[0])
                    if result_iou > 0:
                        try:
                            person_intersection_str = f"Person #{p_bbox[1][0]} interacted with object #{int(box.id[0])} {label_map[int(box.cls[0])]}"
                        except:
                            person_intersection_str = f"Person {p_bbox[1][0]} interacted with object (ID unable to be assigned) {label_map[int(box.cls[0])]}"

                        person_action_str = action_str + \
                            f" by person {p_bbox[1][0]}"
                        return person_action_str

# collect available videos under the data folder in case the user wants to use a different one


def getSampleVideos() -> []:
    files = glob.glob("./data/*.mp4")
    paths = []
    for file in files:
        pfile = Path(file)
        paths.append(pfile)
    return paths

# provide the first video in the folder as a default value


def getFirstSampleVideo() -> Path:
    videos = getSampleVideos()
    if len(videos) >= 1:
        return videos[0]

    return None

# Items is a list of a list with length 2 (row). The row contains 2 elements: <Item:str> and <Quantity:int>


def getPandasDF(items):
    if items is None:
        return None

    if type(items) is list:
        return pd.DataFrame(items, columns=['Item', 'Quantity'])

    if type(items) is dict:
        rows = []
        for k, v in items.items():
            row = []
            row.append(k)
            row.append(v)
            rows.append(row)
        return pd.DataFrame(rows, columns=['Item', 'Quantity'])

    return None

# It adds in the log list a row (dict) with the specific columns


def plog(logtable, message, pclass, pop):
    if message is None or not isinstance(message, six.string_types):
        return

    if pclass is None or not isinstance(pclass, six.string_types):
        return

    if pop is None or not isinstance(pop, six.string_types):
        return

    if logtable is None or not isinstance(logtable, list):
        return

    row = {}
    now = datetime.datetime.now()
    row["time"] = now.strftime('%Y-%m-%dT%H:%M:%S')
    row["class"] = pclass
    row["action"] = pop
    row["message"] = message

    logtable.insert(0, row)

# It will convert a set of text with this organization {'1 #3 bottle', '1 #1 banana', '1 #2 apple'} in a list of rows[item, quantity]


def toList(myset) -> list:
    if myset is None or not isinstance(myset, set):
        return None

    rows = []
    for element in myset:
        row = []
        selement = str(element)
        pattern = r'\w+'
        substrings = re.findall(pattern, selement)

        if len(substrings) >= 3:
            if substrings[0].isdigit():
                row.append(int(substrings[0]))
            else:
                row.append("")

            row.insert(0, ' '.join(substrings[2:]))

            rows.append(row)

    return rows

# Auxiliar function for video processing and UI update


def stream_object_detection(video):
    cap = cv2.VideoCapture(video)

    # This means we will output mp4 videos
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    desired_fps = fps // SUBSAMPLE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    iterating, frame = cap.read()

    n_frames = 0

    # Use UUID to create a unique video file
    output_video_name = f"output_{uuid.uuid4()}.mp4"

    # Output Video
    output_video = cv2.VideoWriter(
        output_video_name, video_codec, desired_fps, (width, height))
    # Log Table
    logtable = []

    batch = []  # Batch  - Frames
    item_list = {}  # Detected Objects

    # Define empty lists to keep track of labels
    original_labels = []
    final_labels = []
    person_bbox = []
    p_items = []
    purchased_items = set(p_items)
    a_items = []
    added_items = set(a_items)
    start = True

    while iterating:  # Go through the frame Sequence
        # Define variables to store interactions that are refreshed per frame
        interactions = []
        person_intersection_str = ""

        if n_frames % SUBSAMPLE == 0:
            batch.append(frame)
        if len(batch) == 2 * desired_fps:
            gr.Info("Detecting...")
            results = model.track(source=batch, show=False, batch=2*desired_fps, verbose=False,
                                  stream=True, stream_buffer=True, persist=True)  # Frames as a arr Images

            for result in results:
                # Obtain predictions from yolov8 model
                frame = result.orig_img
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.class_id < 55]
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                bboxes = result.boxes
                if bboxes.id is not None:  # $
                    detections.tracker_id = bboxes.id.cpu().numpy().astype(int)

                labels = [
                    f'#{tracker_id} {label_map[class_id]} {confidence:0.2f}'
                    for _, _, confidence, class_id, tracker_id
                    in detections
                ]

                # Annotate the frame with the zone and bounding boxes.
                frame = box_annotator.annotate(
                    scene=frame, detections=detections_filtered, labels=labels)
                frame = zone_annotator.annotate(scene=frame)

                objects = [f'#{tracker_id} {label_map[class_id]}' for _,
                           _, confidence, class_id, tracker_id in detections]

                # Accumlate detections by classid
                for _, _, confidence, class_id, tracker_id in detections:
                    if label_map[class_id] in item_list:
                        item_list[label_map[class_id]] += 1
                    else:
                        item_list[label_map[class_id]] = 0

                # If this is the first time we run the application,
                # store the objects' labels as they are at the beginning

                if start:
                    original_labels = objects
                    original_dets = len(detections_filtered)
                    start = False
                else:
                    # To identify if an object has been added or removed
                    # we'll use the original labels and identify any changes
                    final_labels = objects
                    new_dets = len(detections_filtered)
                    # Identify if an object has been added or removed using Counters
                    removed_objects = Counter(
                        original_labels) - Counter(final_labels)
                    added_objects = Counter(
                        final_labels) - Counter(original_labels)

                    # Check for objects being added or removed
                    if new_dets - original_dets != 0 and len(removed_objects) >= 1:
                       # An object has been removed
                        for k, v in removed_objects.items():
                            # For each of the objects, check the IOU between a designated object
                            # and a person.
                            if 'person' not in k:
                                removed_object_str = f"{v} {k} removed from zone"
                                removed_action_str = intersecting_bboxes(
                                    bboxes, person_bbox, removed_object_str)
                                if removed_action_str is not None:
                                    # Add the purchased items to a "receipt" of sorts
                                    if removed_object_str not in purchased_items:
                                        # print(f"{v} {k}", a_items)
                                        # if f"{v} {k}" in a_items:
                                        purchased_items.add(f"{v} {k}")
                                        p_items.append(f" - {v} {k}")
                                # Draw the result on the screen
                                plog(logtable, removed_action_str, k, "remove")

                    if len(added_objects) >= 1:
                        # An object has been added
                        for k, v in added_objects.items():
                            # For each of the objects, check the IOU between a designated object
                            # and a person.
                            if 'person' not in k:
                                added_object_str = f"{v} {k} added to zone"
                                added_action_str = intersecting_bboxes(
                                    bboxes, person_bbox, added_object_str)
                                if added_action_str is not None:
                                    if added_object_str not in added_items:
                                        added_items.add(added_object_str)
                                        a_items.append(added_object_str)

                                # Draw the result on the screen
                                plog(logtable, added_action_str, k, "add")

                output_video.write(frame)
                yield gr.skip(), pd.DataFrame(logtable, columns=['time', 'action', 'class', 'message']), toList(purchased_items), getPandasDF(item_list)

            batch = []  # Restart the batch
            output_video.release()

            # return gr.skip() (instead of a value) will keep the component wihout any change
            yield output_video_name, pd.DataFrame(logtable, columns=['time', 'action', 'class', 'message']), toList(purchased_items), getPandasDF(item_list)

            output_video_name = f"output_{uuid.uuid4()}.mp4"
            output_video = cv2.VideoWriter(
                output_video_name, video_codec, desired_fps, (width, height))  # type: ignore

        iterating, frame = cap.read()
        n_frames += 1

# Open a the browser using the informed URL


def open_browser(url):
    webbrowser.open_new_tab(url)


###########################################################
print("Starting the Automated Self-Checkout Demo...")

# Specify our models path
models_dir = Path("./model")
models_dir.mkdir(exist_ok=True)

print("Loading model...")
DET_MODEL_NAME = "yolov8m"
det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

# Load our Yolov8 object detection model
ov_model_path = Path(
    f"model/{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not ov_model_path.exists():
    # export model to OpenVINO format
    out_dir = det_model.export(format="openvino", dynamic=False, half=True)

model = YOLO("model/yolov8m_openvino_model/", task="detect")

print("Gathering default video and zone information...")
# Default Video
VID_PATH = "data/example.mp4"
video_info = sv.VideoInfo.from_video_path(VID_PATH)
polygon = load_zones("config/zones.json", "test-example-1")
zone = sv.PolygonZone(
    polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

print("Initializing variables...")
# Aux variable to show items and quantity dynamically
SUBSAMPLE = 2  # Video Processing Subsample rate

# Log Table organization
plog_cols = ["time", "class", "action", "message"]
plog_list = []

# Detected Objects
item_list = {}

print("Defining UI...")
header = "# Detect and Track Objects with OpenVINO™ for Self-Checkout\n"
header += "### [Go to Jupyter Notebook](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/automated_self_checkout/self-checkout-recipe.ipynb)"
footer = "**<center>License: [Apache 2.0](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt) | Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) | Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)</center>**"
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown(header)
    with gr.Row(equal_height=True):
        video = gr.PlayableVideo(label="Video Source",
                                 value=getFirstSampleVideo())
        output_video = gr.Video(label="Processed Video",
                                streaming=True, autoplay=True)

    with gr.Row(equal_height=True):
        generate_btn = gr.Button("Start Video Processing", variant="primary")

    with gr.Row(equal_height=True):
        with gr.Column():
            rtlog = gr.DataFrame(
                value=None,
                headers=["time", "action", "class", "message"],
                datatype=["str", "str", "str", "str"],
                label="Detection Message Log",
                column_widths=["25%", "15%", "15%", "45%"],
                show_search="filter",
                show_copy_button=True
            )

        with gr.Column():
            plot = gr.BarPlot(None,  # Empty dataframe
                              x="Item", y="Quantity", y_aggregate="sum",
                              title="Detected Items",
                              height=400  # pixels
                              )
            items = gr.DataFrame(
                value=None,
                headers=["Item", "Added"],
                datatype=["str", "number"],
                label="Purchased Items"
            )
    with gr.Row():
        gr.Markdown(footer)

    generate_btn.click(
        fn=stream_object_detection,
        inputs=[video],
        outputs=[output_video, rtlog, items, plot])

if __name__ == "__main__":
    print("Starting the server...")
    _, localurl, _ = demo.launch(inbrowser=True, prevent_thread_lock=True)
    print("Server running at "+localurl)
    print("Opening browser...")
    try:
        # Timer(1, open_browser(localurl)).start()
        if open_browser(localurl) == False:
            print(
                "Unable to open browser. Please open the following URL in your browser: "+str(localurl))
    except Exception:
        print("Unable to open browser. Please open the following URL in your browser: ")
        print(localurl)

    print("Press Ctrl+C to stop the server")
    try:
        while (True):
            time.sleep(1)
            pass
    except KeyboardInterrupt:
        print("Server stopped")
        pass
