import requests
from moviepy import VideoFileClip


# Browser support for mp4 is better than avi, hence conversion is often necessary
def avi_to_mp4(video_path):
    output_path = video_path.replace(".avi", ".mp4")
    with VideoFileClip(video_path) as clip:
        # These codecs provide good compression and wide compatibility
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_path

def download_video(url, filename):
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in write-binary mode and save the content
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Download complete: {filename}")
    else:
        print("Failed to retrieve the file. HTTP Status Code:", response.status_code)