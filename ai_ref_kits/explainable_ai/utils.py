import hashlib
import shutil
import tempfile
import urllib
import urllib.parse
from os import PathLike
from pathlib import Path


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
    from tqdm.notebook import tqdm_notebook
    import requests

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
        with tqdm_notebook(total=filesize, unit="B", unit_scale=True, disable=not show_progress) as bar:
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