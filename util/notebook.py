
# ideias adaptadas de : https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google
from base64 import b64encode
from IPython.display import HTML, display

from pathlib import Path


def display_video(videopath: str) -> str:
    """
    Gets a string containing a b4-encoded version of the MP4 video
    at the specified path.
    """
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    html_code = f'<video width=400 controls><source src="data:video/mp4;' \
                f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'
    return display(HTML(html_code))


def display_videos_from_path(videos_folder='', prefix=''):
    """
    Adapted from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(videos_folder).glob("{}*.mp4".format(prefix)):
        video_b64 = b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                    </video>'''.format(mp4, video_b64.decode('ascii')))
    display(HTML(data="<br>".join(html)))
