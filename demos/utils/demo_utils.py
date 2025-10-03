import os
import os.path
import threading
import time
import urllib.parse
from os import PathLike
from pathlib import Path
from typing import Tuple, Dict

import numpy as np


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
        import cv2

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
        import cv2
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


def available_devices(exclude: list | tuple | None = None) -> Dict[str, str]:
    import openvino as ov

    exclude_devices = set()
    if exclude is not None:
        exclude_devices.update(exclude)

    device_mapping = {}
    if "AUTO" not in exclude_devices:
        device_mapping["AUTO"] = "Automatic Device Selection"

    core = ov.Core()
    for device in core.available_devices:
        if device not in exclude_devices:
            device_mapping[device] = core.get_property(device, "FULL_DEVICE_NAME")

    return device_mapping


def draw_control_panel(frame: np.ndarray, device_mapping: Dict[str, str], include_precisions: bool = True, include_devices: bool = True) -> None:
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

logo_img = None

def draw_qr_code(frame: np.ndarray, qr_code: np.ndarray) -> None:
    import cv2

    if qr_code.shape[2] != 4:
        qr_code = cv2.cvtColor(qr_code, cv2.COLOR_BGR2BGRA)

    draw_img(frame, qr_code, (frame.shape[1] - qr_code.shape[1], 0), alpha=0.8)


def draw_ov_watermark(frame: np.ndarray, alpha: float = 0.35, size: float = 0.2) -> None:
    import cv2

    global logo_img
    if logo_img is None:
        logo_img = cv2.imread(os.path.join(os.path.dirname(__file__), "assets", "openvino-logo.png"), cv2.IMREAD_UNCHANGED)

    scale = size * frame.shape[1] / logo_img.shape[1]
    watermark = cv2.resize(logo_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    draw_img(frame, watermark, (frame.shape[1] - watermark.shape[1], frame.shape[0] - watermark.shape[0]), alpha)


def draw_img(frame: np.ndarray, img: np.ndarray, point: Tuple[int, int], alpha: float = 1.0) -> None:
    alpha_channel = img[:, :, 3:].astype(float) / 255
    alpha_channel *= alpha
    patch = frame[point[1]:point[1] + img.shape[0], point[0]:point[0] + img.shape[1]]

    patch[:] = alpha_channel * img[:, :, :3] + ((1.0 - alpha_channel) * patch)


def draw_text(image: np.ndarray, text: str, point: Tuple[int, int], center: bool = False, font_scale: float = 1.0, font_color: Tuple[int, int, int] = (255, 255, 255), with_background: bool = False) -> None:
    import cv2

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


def crop_center(image: np.ndarray) -> np.ndarray:
    size = min(image.shape[:2])
    start_x = (image.shape[1] - size) // 2
    start_y = (image.shape[0] - size) // 2
    return image[start_y:start_y + size, start_x:start_x + size]


def get_qr_code(text: str, size: int = 256, with_embedded_image: bool = False) -> np.ndarray:
    import qrcode
    from qrcode.image.styledpil import StyledPilImage
    from qrcode.image.styles.moduledrawers import GappedSquareModuleDrawer
    import PIL.Image

    from PIL import Image, ImageDraw, ImageFont

    get_code_img = None
    # Create a custom image to embed in the QR code
    if with_embedded_image:
        get_code_img = Image.new("RGB", (100, 100), (255, 255, 255))
        draw_context = ImageDraw.Draw(get_code_img)
        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets", "FreeMono.ttf"), 35)
        draw_context.multiline_text((8, 0), "Get\nDemo\nCode", font=font, fill=(0, 0, 0), align="center")

    # Create the QR code
    error_correction = qrcode.constants.ERROR_CORRECT_H if with_embedded_image else qrcode.constants.ERROR_CORRECT_L
    qr = qrcode.QRCode(box_size=10, border=2, error_correction=error_correction)

    qr.add_data(text)
    img = qr.make_image(image_factory=StyledPilImage, module_drawer=GappedSquareModuleDrawer(), embedded_image=get_code_img)

    img = img.resize((size, size), resample=PIL.Image.LANCZOS)
    return np.array(img)


def get_gradio_intel_color(name: str) -> "gr.themes.Color":
    import gradio as gr

    if name == "classic_blue":
        return gr.themes.Color(name="intel_classic_blue", c50="#e4f5ff", c100="#76ceff", c200="#36befe", c300="#00a4f6", c400="#008dd7", c500="#006abb", c600="#005fa7", c700="#004986", c800="#003c6b", c900="#002e54", c950="#001d34")
    elif name == "energy_blue":
        return gr.themes.Color(name="intel_energy_blue", c50="#e2faff",  c100="#b8f3ff", c200="#7bddff", c300="#41d4fb", c400="#11c5f9", c500="#00c7fd", c600="#00addc", c700="#0096ca", c800="#0077a4", c900="#005b85", c950="#003b54")
    else:
        raise ValueError("Unsupported color name")


def gradio_intel_theme() -> "gr.themes.ThemeClass":
    import gradio as gr

    return gr.themes.Base(primary_hue=get_gradio_intel_color("energy_blue"))


def gradio_intel_header(name: str = "") -> "gr.HTML":
    import gradio as gr

    return gr.HTML(
            "<div style='width:100%;max-width:100%;margin-left:0;position:relative;padding:0;box-sizing:border-box;'>"
            "  <div style='margin:0;padding:0 15px;background:#0068bb;height:60px;width:100%;display:flex;align-items:center;position:relative;box-sizing:border-box;margin-bottom:15px;'>"
            f"    <div style='height:60px;line-height:60px;color:white;font-size:24px;'>{name}</div>"
            "    <img src='https://www.intel.com/content/dam/logos/intel-header-logo.svg' style='margin-left:auto;width:60px;height:60px;' />"
            "  </div>"
            "</div>",
        padding=False
    )
