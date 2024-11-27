import os
import os.path
import threading
import time
import urllib.parse
from os import PathLike
from pathlib import Path
from typing import Tuple, Dict

import cv2
import openvino as ov
from numpy import ndarray


def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    show_progress: bool = True,
    silent: bool = False,
    timeout: int = 10,
) -> PathLike:
    """
    Download a file from a url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param show_progress: If True, show an TQDM ProgressBar
    :param silent: If True, do not print a message if the file already exists
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    from tqdm import tqdm
    import requests

    filename = filename or Path(urllib.parse.urlparse(url).path).name
    chunk_size = 16384  # make chunks bigger so that not too many updates are triggered for Jupyter front-end

    filename = Path(filename)
    if len(filename.parts) > 1:
        raise ValueError(
            "`filename` should refer to the name of the file, excluding the directory. "
            "Use the `directory` parameter to specify a target directory for the downloaded file."
        )

    # create the directory if it does not exist, and add the directory to the filename
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / Path(filename)

    try:
        response = requests.get(url=url,
                                headers={"User-agent": "Mozilla/5.0"},
                                stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:  # For error associated with not-200 codes. Will output something like: "404 Client Error: Not Found for url: {url}"
        raise Exception(error) from None
    except requests.exceptions.Timeout:
        raise Exception(
                "Connection timed out. If you access the internet through a proxy server, please "
                "make sure the proxy is set in the shell from where you launched Jupyter."
        ) from None
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}") from None

    # download the file if it does not exist, or if it exists with an incorrect file size
    filesize = int(response.headers.get("Content-length", 0))
    if not filename.exists() or (os.stat(filename).st_size != filesize):

        with tqdm(
            total=filesize,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(filename),
            disable=not show_progress,
        ) as progress_bar:

            with open(filename, "wb") as file_object:
                for chunk in response.iter_content(chunk_size):
                    file_object.write(chunk)
                    progress_bar.update(len(chunk))
                    progress_bar.refresh()
    else:
        if not silent:
            print(f"'{filename}' already exists.")

    response.close()

    return filename.resolve()


class VideoPlayer:
    """
    Custom video player to fulfill FPS requirements. You can set target FPS and output size,
    flip the video horizontally or skip first N frames.

    :param source: Video source. It could be either camera device or video file.
    :param size: Output frame size.
    :param flip: Flip source horizontally.
    :param fps: Target FPS.
    :param skip_first_frames: Skip first N frames.
    """

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(
                f"Cannot open {'camera' if isinstance(source, int) else ''} {source}"
            )
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            self.__cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = (
                cv2.INTER_AREA
                if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                else cv2.INTER_LINEAR
            )
        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

        self.fps = self.__input_fps
        self.width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    """
    Start playing.
    """

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    """
    Stop playing and release resources.
    """

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    """
    Get current frame.
    """

    def next(self):
        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = cv2.flip(frame, 1)
        return frame


logo_img = cv2.imread(os.path.join(os.path.dirname(__file__), "openvino-logo.png"), cv2.IMREAD_UNCHANGED)


def available_devices() -> Dict[str, str]:
    device_mapping = {"AUTO": "AUTO device"}

    core = ov.Core()
    for device in core.available_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        if "nvidia" not in device_name.lower():
            device_mapping[device] = device_name

    return device_mapping


def draw_control_panel(frame: ndarray, device_mapping: Dict[str, str], include_precisions: bool = True, include_devices: bool = True) -> None:
    h, w = frame.shape[:2]
    line_space = 40
    start_y = h - (include_devices * len(device_mapping) + include_precisions * 2 + 1) * line_space - 20
    draw_text(frame, "Control panel. Press:", (10, start_y))
    next_y = start_y + line_space
    if include_precisions:
        draw_text(frame, "f: FP16 model", (10, next_y))
        draw_text(frame, "i: INT8 model", (10, next_y + line_space))
        next_y += 2 * line_space
    if include_devices:
        for i, (device_name, device_info) in enumerate(device_mapping.items(), start=1):
            draw_text(frame, f"{i}: {device_name} - {device_info}", (10, next_y))
            next_y += line_space


def draw_ov_watermark(frame: ndarray, alpha: float = 0.35, size: float = 0.2) -> None:
    scale = size * frame.shape[1] / logo_img.shape[1]
    watermark = cv2.resize(logo_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    alpha_channel = watermark[:, :, 3:].astype(float) / 255
    alpha_channel *= alpha
    patch = frame[frame.shape[0] - watermark.shape[0]:, frame.shape[1] - watermark.shape[1]:]

    patch[:] = alpha_channel * watermark[:, :, :3] + ((1.0 - alpha_channel) * patch)


def draw_text(image: ndarray, text: str, point: Tuple[int, int], center: bool = False, font_scale: float = 1.0, font_color: Tuple[int, int, int] = (255, 255, 255), with_background: bool = False) -> None:
    _, f_width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale * f_width / 2000, thickness=2)

    rect_width = text_size[0] + 50
    rect_height = text_size[1] + 30
    rect_x, rect_y = (point[0] - rect_width // 2, point[1] - rect_height // 2) if center else point

    if with_background:
        cv2.rectangle(image, pt1=(rect_x, rect_y), pt2=(rect_x + rect_width, rect_y + rect_height), color=(0, 0, 0), thickness=cv2.FILLED)

    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2

    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale * f_width / 2000, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale * f_width / 2000, color=font_color, thickness=1, lineType=cv2.LINE_AA)
