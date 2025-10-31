import hashlib
import os
import shutil
import tempfile
import threading
import time
import requests
import urllib.parse
from os import PathLike
from pathlib import Path
import cv2
import tqdm


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


def download_file(
    url: PathLike,
    filename: PathLike = None,
    directory: PathLike = None,
    expected_hash: str = None,
    show_progress: bool = True,
    timeout: int = 10,
) -> Path:
    """
    Download a file from an url and save it to the local filesystem. The file is saved to the
    current directory by default, or to `directory` if specified. If a filename is not given,
    the filename of the URL will be used.

    :param url: URL that points to the file to download
    :param filename: Name of the local file to save. Should point to the name of the file only,
                     not the full path. If None the filename from the url will be used
    :param directory: Directory to save the file to. Will be created if it doesn't exist
                      If None the file will be saved to the current working directory
    :param expected_hash: Expected hash of the file to verify integrity. If None, no hash check is performed
    :param show_progress: If True, show an TQDM ProgressBar
    :param timeout: Number of seconds before cancelling the connection attempt
    :return: path to downloaded file
    """
    filename = filename or Path(urllib.parse.urlparse(url).path).name
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = Path(directory) / filename
    else:
        filename = Path(filename)

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    # Download to temporary file
    filesize = int(response.headers.get("Content-Length", 0))
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        with tqdm.tqdm(total=filesize, unit="B", unit_scale=True, disable=not show_progress) as bar:
            for chunk in response.iter_content(16384):
                tmp_file.write(chunk)
                bar.update(len(chunk))
        tmp_path = Path(tmp_file.name)

    # Verify hash if provided
    if expected_hash:
        hasher = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        if hasher.hexdigest().lower() != expected_hash.lower():
            tmp_path.unlink(missing_ok=True)
            raise ValueError("Hash mismatch — file integrity verification failed.")

    # Move only after validation passes
    shutil.move(tmp_path, filename)
    return filename.resolve()
