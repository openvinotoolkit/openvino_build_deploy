from utils import demo_utils as utils
import cv2
import numpy as np
import requests
from openvino import Core
import logging
from pathlib import Path
import time
import traceback
import collections
import os
import sys
from plyer import notification  # Import plyer for notifications

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
SCRIPT_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "..",
    "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))


def download_model(model_name, precision):
    """Download the required OpenVINO model if not already present."""
    base_model_dir = Path("model")
    model_dir = base_model_dir / "intel" / model_name / precision
    model_path = model_dir / f"{model_name}.xml"
    bin_path = model_dir / f"{model_name}.bin"

    if not model_path.exists() or not bin_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        xml_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/{model_name}.xml"
        bin_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/{model_name}.bin"

        logging.info(f"Downloading model from {xml_url}")
        with requests.get(xml_url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info(f"Downloading model from {bin_url}")
        with requests.get(bin_url, stream=True) as r:
            r.raise_for_status()
            with open(bin_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    logging.info(f"Using model at {model_path}")
    return model_path


def load_model(model_path, device):
    """Load the OpenVINO model."""
    core = Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)

    input_layer = compiled_model.input(0)
    output_layers = [compiled_model.output(i) for i in range(
        len(compiled_model.outputs))]  # Handle multiple outputs

    logging.info(f"Loaded model: {model_path}")
    logging.info(f"Input layer shape: {input_layer.shape}")
    for i, output_layer in enumerate(output_layers):
        logging.info(f"Output layer {i} shape: {output_layer.shape}")

    return compiled_model, input_layer, output_layers


def preprocess_head_pose_input(frame, keypoints):
    """Extract the face region and preprocess it for model input."""
    face_region = extract_face_region(frame, keypoints)

    if face_region is None or face_region.shape[0] == 0 or face_region.shape[1] == 0:
        logging.error("Invalid face region extracted. Skipping frame.")
        return None

    input_img = cv2.resize(face_region, (60, 60))  # Resize to (60, 60)
    input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...].astype(
        np.float32)  # Shape: (1, 3, 60, 60)

    logging.debug(f"Preprocessed input shape: {input_img.shape}")
    return input_img


def extract_face_region(frame, keypoints):
    """Extract the face region using detected keypoints."""
    try:
        x_min = int(np.min(keypoints[:, 0]))
        y_min = int(np.min(keypoints[:, 1]))
        x_max = int(np.max(keypoints[:, 0]))
        y_max = int(np.max(keypoints[:, 1]))

        # Ensure coordinates are within frame boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Check if the region is valid
        if x_max <= x_min or y_max <= y_min:
            logging.error("Invalid face region coordinates.")
            return None

        return frame[y_min:y_max, x_min:x_max]
    except Exception as e:
        logging.error(f"Error extracting face region: {e}")
        return None


def check_neck_posture(
        yaw,
        pitch,
        roll,
        feedback_buffer,
        feedback_interval=2,
        buffer_duration=10):
    """Check for poor neck posture and return feedback."""
    current_time = time.time()
    if feedback_buffer and current_time - \
            feedback_buffer[0][1] > buffer_duration:
        feedback_buffer.popleft()  # Remove old feedbacks

    yaw_threshold = 20  # Adjusted threshold for head turning too far left or right
    pitch_threshold = 25  # Adjusted threshold for head tilted too far downward
    roll_threshold = 30  # Adjusted threshold for head tilting sideways too much

    feedback = "Good posture maintained."
    poor_posture = False

    if abs(yaw) > yaw_threshold or abs(roll) > roll_threshold:
        poor_posture = True
        feedback = "You're turning your head too much! Sit up straight!"
    elif abs(pitch) > pitch_threshold:
        poor_posture = True
        feedback = "You're slouching too much! Sit up straight!"

    feedback_buffer.append((feedback, current_time, poor_posture))

    # Log yaw, pitch, roll values and feedback buffer
    logging.info(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")
    logging.info(f"Feedback: {feedback}, Poor Posture: {poor_posture}")

    # Determine the most critical feedback
    poor_posture_duration = sum(1 for fb, _, pp in feedback_buffer if pp)
    if poor_posture_duration > feedback_interval:
        return feedback, current_time, poor_posture

    return "Good posture maintained.", current_time, False


def draw_pose_lines(frame, yaw, pitch, roll, keypoints):
    """Draw lines on the face to visualize the detected angles."""
    center = np.mean(keypoints, axis=0).astype(int)
    length = 50

    # Calculate end points for the lines
    yaw_end = (int(center[0] + length * np.sin(np.radians(yaw))),
               int(center[1] - length * np.cos(np.radians(yaw))))
    pitch_end = (int(center[0] +
                     length *
                     np.sin(np.radians(pitch))), int(center[1] +
                                                     length *
                                                     np.cos(np.radians(pitch))))
    roll_end = (int(center[0] + length * np.cos(np.radians(roll))),
                int(center[1] + length * np.sin(np.radians(roll))))

    # Draw lines with different lengths and directions
    cv2.line(frame, tuple(center), yaw_end, (0, 255, 0), 2)  # Green for yaw
    cv2.line(frame, tuple(center), pitch_end, (255, 0, 0), 2)  # Blue for pitch
    cv2.line(frame, tuple(center), roll_end, (0, 0, 255), 2)  # Red for roll

    # Draw text for yaw, pitch, roll in the center of the face
    cv2.putText(frame, f"Y: {yaw:.2f}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.putText(frame, f"P: {pitch:.2f}",
                (center[0] + 10, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)
    cv2.putText(frame, f"R: {roll:.2f}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    return frame


def detect_keypoints(frame, model, input_layer, output_layer):
    """Detect keypoints using the facial landmark model."""
    if frame is None or frame.size == 0:
        logging.error("Empty frame passed to detect_keypoints.")
        return None

    input_img = cv2.resize(frame, (input_layer.shape[2], input_layer.shape[3]))
    input_img = input_img.transpose(
        2, 0, 1)[
        np.newaxis, ...].astype(
            np.float32)  # Shape: (1, 3, H, W)

    # Access the first (and only) output layer
    results = model([input_img])[output_layer[0]]

    # Reshape to (35, 2) since we have 35 (x, y) keypoints
    keypoints = results.reshape(35, 2)

    # Scale keypoints to the original frame size
    keypoints[:, 0] *= frame.shape[1] / input_layer.shape[2]
    keypoints[:, 1] *= frame.shape[0] / input_layer.shape[3]

    return keypoints


def detect_faces(frame, model, input_layer, output_layer):
    """Detect faces using the face detection model."""
    input_img = cv2.resize(
        frame,
        (input_layer.shape[3],
         input_layer.shape[2]))  # Resize to (672, 384)
    input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...].astype(
        np.float32)  # Shape: (1, 3, 384, 672)

    # Access the first (and only) output layer
    results = model([input_img])[output_layer[0]]

    # Extract bounding boxes
    boxes = []
    for result in results[0][0]:
        if result[2] > 0.5:  # Confidence threshold
            xmin = int(result[3] * frame.shape[1])
            ymin = int(result[4] * frame.shape[0])
            xmax = int(result[5] * frame.shape[1])
            ymax = int(result[6] * frame.shape[0])
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])

    return boxes


def run_demo(
        source,
        face_detection_model_name,
        head_pose_model_name,
        keypoint_model_name,
        model_precision,
        device,
        flip):
    """Run the head pose detection demo."""
    cap = None
    feedback_buffer = collections.deque()  # Initialize feedback buffer
    processing_times = collections.deque()  # Initialize processing times deque

    try:
        # Load models
        face_detection_model_path = download_model(
            face_detection_model_name, model_precision)
        face_detection_model, face_detection_input, face_detection_output = load_model(
            face_detection_model_path, device)

        head_pose_model_path = download_model(
            head_pose_model_name, model_precision)
        head_pose_model, head_pose_input, head_pose_outputs = load_model(
            head_pose_model_path, device)

        keypoint_model_path = download_model(
            keypoint_model_name, model_precision)
        keypoint_model, keypoint_input, keypoint_output = load_model(
            keypoint_model_path, device)

        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)

        if not cap.isOpened():
            logging.error("Failed to open video source.")
            return

        # Set window properties
        cv2.namedWindow("Head Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Head Pose Estimation",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream.")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Measure processing time
            start_time = time.time()

            # Detect faces
            boxes = detect_faces(
                frame,
                face_detection_model,
                face_detection_input,
                face_detection_output)

            for box in boxes:
                x, y, w, h = box
                face_frame = frame[y:y + h, x:x + w]

                # Detect keypoints
                keypoints = detect_keypoints(
                    face_frame, keypoint_model, keypoint_input, keypoint_output)

                if keypoints is None or keypoints.shape[0] == 0:
                    logging.error("No keypoints detected.")
                    continue

                # Preprocess input for head pose model
                input_img = preprocess_head_pose_input(face_frame, keypoints)

                if input_img is None:
                    continue  # Skip frame if preprocessing failed

                # Run inference on head pose model
                yaw = head_pose_model([input_img])[head_pose_outputs[0]]
                pitch = head_pose_model([input_img])[head_pose_outputs[1]]
                roll = head_pose_model([input_img])[head_pose_outputs[2]]

                # Extract yaw, pitch, roll
                yaw, pitch, roll = yaw[0][0], pitch[0][0], roll[0][0]
                posture_feedback, _, poor_posture = check_neck_posture(
                    yaw, pitch, roll, feedback_buffer)

                if posture_feedback:
                    feedback = posture_feedback
                    logging.info(posture_feedback)
                    color = (
                        0,
                        255,
                        0) if feedback == "Good posture maintained." else (
                        0,
                        0,
                        255)
                    cv2.putText(
                        frame,
                        posture_feedback,
                        (x,
                         y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)  # Reduced font size, not bold
                    # Show notification if posture is not good
                    if feedback != "Good posture maintained.":
                        notification.notify(
                            title="Posture Alert",
                            message=feedback,
                            timeout=5
                        )

                # Draw pose lines
                face_frame = draw_pose_lines(
                    face_frame, yaw, pitch, roll, keypoints)

                # Replace the face region in the original frame
                frame[y:y + h, x:x + w] = face_frame

                # Draw bounding box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Measure processing time
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames
            if len(processing_times) > 200:
                processing_times.popleft()

            # Calculate mean processing time and FPS
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            utils.draw_text(
                frame,
                text=f"Currently running models ({model_precision}) on {device}",
                point=(
                    10,
                    10))
            utils.draw_text(
                frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (10, 50))

            # Display result
            cv2.imshow("Head Pose Estimation", frame)

            # Introduce a delay to slow down the processing
            time.sleep(0.2)  # Adjust the delay as needed

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        logging.error(traceback.format_exc())
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str,
                        help="Video file path or webcam ID")
    parser.add_argument(
        '--device',
        default="AUTO",
        type=str,
        help="Device to start inference on")
    parser.add_argument(
        '--face_detection_model_name',
        type=str,
        default="face-detection-adas-0001",
        help="Face detection model to be used")
    parser.add_argument(
        '--head_pose_model_name',
        type=str,
        default="head-pose-estimation-adas-0001",
        help="Head pose estimation model to be used")
    parser.add_argument(
        '--keypoint_model_name',
        type=str,
        default="facial-landmarks-35-adas-0002",
        help="Keypoint detection model to be used")
    parser.add_argument(
        '--model_precision',
        type=str,
        default="FP32",
        choices=[
            "FP16-INT8",
            "FP16",
            "FP32"],
        help="Model precision")
    parser.add_argument(
        '--flip',
        type=bool,
        default=True,
        help="Flip input video")

    args = parser.parse_args()
    run_demo(
        args.stream,
        args.face_detection_model_name,
        args.head_pose_model_name,
        args.keypoint_model_name,
        args.model_precision,
        args.device,
        args.flip)