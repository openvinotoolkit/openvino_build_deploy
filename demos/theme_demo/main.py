import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

import themes

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils


def load_theme(theme: str, device: str):
    if theme == "christmas":
        return themes.ChristmasTheme(device)
    elif theme == "halloween":
        return themes.HalloweenTheme(device)
    elif theme == "easter":
        return themes.EasterTheme(device)
    else:
        raise ValueError(f"Unknown theme: {theme}")


def run_demo(source: str, theme: str, device: str, flip: bool = True):
    device_mapping = utils.available_devices()

    theme_obj = load_theme(theme, device)

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

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break

            # Measure processing time.
            start_time = time.time()

            detections = theme_obj.run_inference(frame)

            stop_time = time.time()

            # Draw watermark
            utils.draw_ov_watermark(frame)

            # Draw boxes on a frame.
            frame = theme_obj.draw_results(frame, detections)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            utils.draw_text(frame, text=f"Currently running models ({theme_obj.model_precision}) on {theme_obj.device}", point=(10, 10))
            utils.draw_text(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (10, 50))

            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)

            # escape = 27 or 'q' to close the app
            if key == 27 or key == ord('q'):
                break

            for i, dev in enumerate(device_mapping.keys()):
                if key == ord('1') + i:
                    theme_obj.load_models(dev)
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
    parser.add_argument('--device', default="CPU", type=str, help="Device to start inference on")
    parser.add_argument("--theme", type=str, default="easter", choices=["christmas", "halloween", "easter"], help="Theme to be used")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run_demo(args.stream, args.theme, args.device, args.flip)