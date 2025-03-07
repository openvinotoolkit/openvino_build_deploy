import argparse
import collections
import os
import sys
import time
import openvino as ov
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


EMOTION_CLASSES = ["neutral", "happy", "sad", "surprise", "anger"]
EMOTION_MAPPING = {"neutral": "Rudolph", "happy": "Cupid", "surprise": "Blitzen", "sad": "Prancer", "anger": "Vixen"}

santa_beard_img = cv2.imread("assets/santa_beard.png", cv2.IMREAD_UNCHANGED)
santa_cap_img = cv2.imread("assets/santa_cap.png", cv2.IMREAD_UNCHANGED)
reindeer_nose_img = cv2.imread("assets/reindeer_nose.png", cv2.IMREAD_UNCHANGED)
reindeer_sunglasses_img = cv2.imread("assets/reindeer_sunglasses.png", cv2.IMREAD_UNCHANGED)
reindeer_antlers_img = cv2.imread("assets/reindeer_antlers.png", cv2.IMREAD_UNCHANGED)


def download_model(model_name, precision, provider="intel", suffix='xml'):
    base_model_dir = Path("model")

    model_path = base_model_dir / f"{provider}" / model_name / precision / f"{model_name}.{suffix}"

    if provider == "intel" and not model_path.exists():
        model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        utils.download_file(model_url_dir + model_name + '.bin', model_path.with_suffix('.bin').name, model_path.parent)
        utils.download_file(model_url_dir + model_name + '.xml', model_path.name, model_path.parent)

    if provider == "google" and not model_path.exists():
        model_url_dir = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/{model_name}/float32/latest/"
        utils.download_file(model_url_dir + model_name + '.tflite', model_path.name, model_path.parent)

    return model_path


def load_model(model_path, device):
    # Initialize OpenVINO Runtime.
    core = Core()

    # Read the network and corresponding weights from a file.
    ir_model_path = model_path.with_suffix(".xml")

    if not ir_model_path.exists():
        # Convert to IR
        model = ov.convert_model(model_path)
        ov.save_model(model, ir_model_path)
    else:
        model = core.read_model(ir_model_path)

    # Compile the model for CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = core.compile_model(model=model, device_name=device)

    # Get the input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer


def preprocess_images(imgs, width, height):
    result = []
    for img in imgs:
        # Resize the image and change dims to fit neural network input.
        input_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0)
        result.append(input_img)
    return np.array(result)


def process_instance_segmentation_results(frame, bg_image, results, in_width, in_height, thresh=0.5):
    labels = results['labels']
    boxes = results['boxes']
    masks = results['masks']

    h, w = frame.shape[:2]
    scale_x, scale_y = w / in_width, h / in_height

    valid_indices = np.where(
        (labels == 0) & # Default person class label is 0
        (boxes[:, 4] > thresh) # Confidence score above threshold
    )[0]

    foreground_mask = np.zeros((h, w), dtype=np.uint8)

    for idx in valid_indices:
        xmin, ymin, xmax, ymax, _ = boxes[idx]

        # Scale coordinates
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        
        # Clip coordinates
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        
        if xmax <= xmin or ymax <= ymin:
            continue

        # Process foreground mask
        mask = cv2.resize(masks[idx], (xmax - xmin, ymax - ymin))
        mask_binary = ((mask > thresh) * 255).astype(np.uint8)
        
        foreground_mask[ymin:ymax, xmin:xmax] = cv2.bitwise_or(
            foreground_mask[ymin:ymax, xmin:xmax],
            mask_binary
        )

    # Background replacement
    background_mask = cv2.bitwise_not(foreground_mask)
    bg_resized = cv2.resize(bg_image, (w, h))
    background_mask = cv2.merge([background_mask] * 3)
    return np.where(background_mask == 0, frame, bg_resized)


def process_selfie_segmentation_results(frame, bg_image, results):
    h, w = frame.shape[:2]
    background_mask = np.argmax(results[0], -1)[0] # background class label is 0
    background_mask = cv2.resize(background_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    background_mask = cv2.merge([background_mask] * 3)
    frame = np.where(background_mask  > 0, frame, bg_image)
    return frame


def process_detection_results(frame, results, in_width, in_height, thresh=0.5):
    # The size of the original frame.
    h, w = frame.shape[:2]
    scale_x, scale_y = w / in_width, h / in_height
    # The 'results' variable is a [200, 5] tensor.
    results = results.squeeze()
    boxes = []
    scores = []
    for xmin, ymin, xmax, ymax, score in results:
        # Create a box with pixels real coordinates from the output box.
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        boxes.append(tuple(map(int, (xmin * scale_x, ymin * scale_y, (xmax - xmin) * scale_x, (ymax - ymin) * scale_y))))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(scores[idx], boxes[idx]) for idx in indices.flatten()]


def process_landmark_results(boxes, results):
    landmarks = []

    for box, result in zip(boxes, results):
        # create a vector of landmarks (35x2)
        result = result.reshape(-1, 2)
        box = box[1]
        # move every landmark according to box origin
        landmarks.append((result * box[2:] + box[:2]).astype(np.int32))

    return landmarks


def draw_mask(img, mask_img, center, face_size, scale=1.0, offset_coeffs=(0.5, 0.5)):
    face_width, face_height = face_size

    # scale mask to fit face size
    mask_width = max(1.0, face_width * scale)
    f_scale = mask_width / mask_img.shape[1]
    mask_img = cv2.resize(mask_img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_AREA)

    x_offset_coeff, y_offset_coeff = offset_coeffs

    # left-top and right-bottom points
    x1, y1 = center[0] - int(mask_img.shape[1] * x_offset_coeff), center[1] - int(mask_img.shape[0] * y_offset_coeff)
    x2, y2 = x1 + mask_img.shape[1], y1 + mask_img.shape[0]

    # if points inside image
    if 0 < x2 < img.shape[1] and 0 < y2 < img.shape[0] or 0 < x1 < img.shape[1] and 0 < y1 < img.shape[1]:
        # face image to be overlayed
        face_crop = img[max(0, y1):min(y2, img.shape[0]), max(0, x1):min(x2, img.shape[1])]
        # overlay
        mask_img = mask_img[max(0, -y1):max(0, -y1) + face_crop.shape[0], max(0, -x1):max(0, -x1) + face_crop.shape[1]]
        # alpha channel to blend images
        alpha_pumpkin = mask_img[:, :, 3:4] / 255.0
        alpha_bg = 1.0 - alpha_pumpkin

        # blend images
        face_crop[:] = (alpha_pumpkin * mask_img)[:, :, :3] + alpha_bg * face_crop


def draw_santa(img, detection):
    (score, box), landmarks, emotion = detection
    # draw beard
    draw_mask(img, santa_beard_img, landmarks[5], box[2:], offset_coeffs=(0.5, 0))
    # draw cap
    draw_mask(img, santa_cap_img, np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.5, offset_coeffs=(0.56, 0.78))


def draw_reindeer(img, landmarks, box):
    # draw antlers
    draw_mask(img, reindeer_antlers_img, np.mean(landmarks[13:17], axis=0, dtype=np.int32), box[2:], scale=1.8, offset_coeffs=(0.5, 1.1))
    # draw sunglasses
    draw_mask(img, reindeer_sunglasses_img, np.mean(landmarks[:4], axis=0, dtype=np.int32), box[2:], offset_coeffs=(0.5, 0.33))
    # draw nose
    draw_mask(img, reindeer_nose_img, landmarks[4], box[2:], scale=0.25)


def draw_christmas_masks(frame, detections):
    # sort by face size
    detections = list(sorted(detections, key=lambda x: x[0][1][2] * x[0][1][3]))

    if not detections:
        return frame

    # others are reindeer
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

    # the largest face is santa
    draw_santa(frame, detections[-1])

    return frame


def run_demo(source, face_detection_model, face_landmarks_model, face_emotions_model, segmentation_model, model_precision, device, flip):
    device_mapping = utils.available_devices()

    face_detection_model_path = download_model(face_detection_model, model_precision)
    face_landmarks_model_path = download_model(face_landmarks_model, model_precision)
    face_emotions_model_path = download_model(face_emotions_model, model_precision)
    
    if segmentation_model.startswith('instance-segmentation'):
        segmentation_model_path = download_model(segmentation_model, model_precision)
    elif segmentation_model == 'selfie_multiclass_256x256':
        segmentation_model_path = download_model(segmentation_model, "FP32", provider="google", suffix="tflite")
    else:
        raise RuntimeError(f'Model {segmentation_model} is not supported.')

    # load face detection model
    fd_model, fd_input, fd_output = load_model(face_detection_model_path, device)
    fd_height, fd_width = list(fd_input.shape)[2:4]

    # load face landmarks model
    fl_model, fl_input, fl_output = load_model(face_landmarks_model_path, device)
    fl_height, fl_width = list(fl_input.shape)[2:4]

    # load emotion classification model
    fe_model, fe_input, fe_output = load_model(face_emotions_model_path, device)
    fe_height, fe_width = list(fe_input.shape)[2:4]
    
    # load segmentation model
    seg_model, seg_input, _ = load_model(segmentation_model_path, device)
    if segmentation_model == 'selfie_multiclass_256x256':
        # input shape is 1*256*256*3 (N, H, W, C)
        seg_height, seg_width = list(seg_input.shape)[1:3]
    elif segmentation_model.startswith('instance-segmentation'):
        seg_height, seg_width = list(seg_input.shape)[2:4]

    
    def replace_background(img, bg_image):
        input_img = preprocess_images([img], seg_width, seg_height)[0]
        if segmentation_model == 'selfie_multiclass_256x256':
            input_img = input_img.transpose(0, 2, 3 ,1) # N, C, H, W -> N, H, W, C
            input_img = input_img.astype(np.float32) / 255 # normalize
            results = seg_model([input_img])
            return process_selfie_segmentation_results(img, bg_image, results)
        elif segmentation_model.startswith('instance-segmentation'):
            results = seg_model([input_img])
            return process_instance_segmentation_results(img, bg_image, results, seg_width, seg_height, thresh=0.5)

    def detect_faces(img):
        input_img = preprocess_images([img], fd_width, fd_height)[0]
        results = fd_model([input_img])[fd_output]
        return process_detection_results(img, results, fd_width, fd_height, thresh=0.25)

    def detect_landmarks(img, boxes):
        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fl_width, fl_height)
        # there are many faces on the image
        results = [fl_model([patch])[fl_output].squeeze() for patch in patches]
        return process_landmark_results(boxes, results)

    def recognize_emotions(img, boxes):
        # every patch is a face image
        patches = [img[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :] for _, box in boxes]
        patches = preprocess_images(patches, fe_width, fe_height)
        # there are many faces on the image
        results = [fe_model([patch])[fe_output].squeeze() for patch in patches]

        if not results:
            return []

        # map result to labels
        labels = list(map(lambda x: EMOTION_CLASSES[x], np.argmax(results, axis=1)))
        return labels

    player = None
    try:
        if isinstance(source, str) and source.isnumeric():
            source = int(source)
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source=source, flip=flip, size=(1920, 1080), fps=30)
        # Start capturing.
        player.start()
        title = "Press ESC to Exit"
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        bg_image = cv2.imread("assets/christmas_background.jpg")
        virtual_background = False

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            # Measure processing time.
            start_time = time.time()

            boxes = detect_faces(frame)
            landmarks = detect_landmarks(frame, boxes)
            emotions = recognize_emotions(frame, boxes)
            detections = zip(boxes, landmarks, emotions)
            
            if virtual_background:
                if bg_image.shape != frame.shape:
                    bg_image = cv2.resize(bg_image, frame.shape[:2][::-1], interpolation=cv2.INTER_AREA)
                frame = replace_background(frame, bg_image)
                    
            stop_time = time.time()

            # Draw watermark
            utils.draw_ov_watermark(frame)

            # Draw boxes on a frame.
            frame = draw_christmas_masks(frame, detections)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            utils.draw_text(frame, text=f"Currently running models ({model_precision}) on {device}", point=(10, 10))
            utils.draw_text(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (10, 50))

            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)

            # escape = 27 or 'q' to close the app
            if key == 27 or key == ord('q'):
                break

            # 'b' to switch background
            if key == ord('b'):
                virtual_background = not virtual_background

            for i, dev in enumerate(device_mapping.keys()):
                if key == ord('1') + i:
                    del fd_model, fl_model, fe_model
                    fd_model, fd_input, fd_output = load_model(face_detection_model_path, dev)
                    fl_model, fl_input, fl_output = load_model(face_landmarks_model_path, dev)
                    fe_model, fe_input, fe_output = load_model(face_emotions_model_path, dev)
                    device = dev
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
    parser.add_argument('--device', default="AUTO", type=str, help="Device to start inference on")
    parser.add_argument("--detection_model_name", type=str, default="face-detection-0205", help="Face detection model to be used")
    parser.add_argument("--landmarks_model_name", type=str, default="facial-landmarks-35-adas-0002", help="Face landmarks regression model to be used")
    parser.add_argument("--emotions_model_name", type=str, default="emotions-recognition-retail-0003", help="Face emotions recognition model to be used")
    parser.add_argument("--segmentation_model_name", type=str, default="instance-segmentation-security-1039", 
                        choices=["instance-segmentation-person-0007", "instance-segmentation-security-1039", 
                                 "instance-segmentation-security-1040", "selfie_multiclass_256x256"],
                        help="Instance segmentation model to be used")
    parser.add_argument("--model_precision", type=str, default="FP16-INT8", choices=["FP16-INT8", "FP16", "FP32"], help="All models precision")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_demo(args.stream, args.detection_model_name, args.landmarks_model_name, args.emotions_model_name, args.segmentation_model_name,
             args.model_precision, args.device, args.flip)