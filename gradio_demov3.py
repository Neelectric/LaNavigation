import gradio as gr

# Function that returns an HTML audio element
def play_audio():
    audio_html = """
    <audio controls>
      <source src="welcome.mp3" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>
    """
    return audio_html

# Create the Gradio interface
app = gr.Interface(
    fn=play_audio, 
    inputs=[], 
    outputs="html", 
    title="Audio Player",
    description="Press the button to play audio."
)

import os

# Get the current working directory
directory_path = os.getcwd()

# List all files in the current directory
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        print(filename)

# Launch the app
app.launch()
