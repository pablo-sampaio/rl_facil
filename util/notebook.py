
# ideias adaptadas de : https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google
from base64 import b64encode
from IPython.display import HTML, display, clear_output

from pathlib import Path


def display_video(videopath: str) -> str:
    """
    Displays a single video file in a notebook.
    """
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    html_code = f'<video width=400 controls><source src="data:video/mp4;' \
                f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'
    return display(HTML(html_code))


'''
def display_videos_from_path(videos_folder='', prefix=''):
    " ""
    Adapted from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    " ""
    html = []
    for mp4 in Path(videos_folder).glob("{}*.mp4".format(prefix)):
        video_b64 = b64encode(mp4.read_bytes())
        html.append(' ''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                    </video>'' '.format(mp4, video_b64.decode('ascii')))
    display(HTML(data="<br>".join(html)))
'''

# by chatgpt
def display_videos_from_path(video_folder='', prefix='rl-video', speed=1.0):
    """
    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) To filter the video files
    """
    html = []
    for mp4 in Path(video_folder).glob("{}*.mp4".format(prefix)):
        video_b64 = b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                    </video>'''.format(mp4, video_b64.decode('ascii')))
    display(HTML(data="<br>".join(html)))
    
    # Add JavaScript code to set playback speed
    display(HTML(f'''
        <script>
        var videos = document.getElementsByTagName('video');
        for (var i = 0; i < videos.length; i++) {{
            videos[i].playbackRate = {speed};
        }}
        </script>
        '''))


def display_videos_from_path_widgets(videos_folder='', prefix=''):
    """
    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) To filter the video files
    """
    import ipywidgets as widgets

    # Function to handle button clicks and display the corresponding video
    def _display_video(index):
        clear_output()
        display(button_box)  # Display the button box again
        print(f"Playing {video_paths[index]}")
        display(video_widgets[index])

    video_paths = []
    video_widgets = []
    for mp4 in Path(videos_folder).glob("{}*.mp4".format(prefix)):
        # Define the video paths / Create video widgets for each video
        video_paths.append(mp4)
        video_widgets.append(widgets.Video.from_file(mp4, autoplay=True, controls=True, loop=True))

    # Create a list of buttons for each video widget
    # Assign the corresponding button click to each video widget
    buttons = []
    for i in range(len(video_widgets)):
        btn = widgets.Button(description=f'Video {i}')
        btn.on_click(lambda event, index=i: _display_video(index))
        buttons.append(btn)

    # Create a horizontal box to display the buttons
    button_box = widgets.HBox(buttons)

    # Display the button box and the initial video
    display(button_box)
    _display_video(0)
