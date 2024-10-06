from io import BytesIO
import queue
import os
from typing import List, Optional
from lavague.core import  WorldModel, ActionEngine
from lavague.core.agents import WebAgent
import gradio as gr
from PIL import Image
from lavague.contexts.gemini import GeminiContext
from lavague.drivers.selenium import SeleniumDriver
import numpy as np
from transformers import pipeline

from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from lavague.core.context import Context
from src.load_config import load_key, load_key_Titas
from src.pixtral_wrapper import PixtralWrapper
from lavague.drivers.selenium import SeleniumDriver
from lavague.core import WorldModel, ActionEngine, PythonEngine
from llama_index.multi_modal_llms.mistralai import MistralAIMultiModal

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0)

def spacebar_pressed(name):
    return f"Hello, {name}! You pressed the spacebar!"

def enter_pressed(name):
    return f"Hello, {name}! You pressed the enter key!"

from pyt2s.services import stream_elements
from pydub import AudioSegment
from pydub.playback import play

def generate_and_play_tts(text, filename='welcome.mp3'):
    # Request TTS data from the stream_elements service
    data = stream_elements.requestTTS(text)

    # Write the audio data to an MP3 file
    with open(filename, 'wb') as file:
        file.write(data)

    return filename

def transcribe(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return "https://www." + transcriber({"sampling_rate": sr, "raw": y})["text"].strip(".").replace(" ", "")

def transcribe_objective(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

class GradioAgentDemo:
    """
    Launch an agent gradio demo of lavague

    Args:
        driver (`BaseDriver`):
            The driver
        context (`ActionContext`):
            An action context
    """

    html = """
    <div class="list-container">
        <ul>
        </ul>
    </div>
    """

    css = """
        .my-button {
            max-height: 3rem !important;
            max-width: 5rem !important;
            min-width: min(100px,100%) !important;
    }
    """

    title = """
    <div class='parent' align="center">
    <div class='child' style="display: inline-block !important;">
    <div style="display: flex; align-items: center;">
    <img src="/file/announcing-mistral.png" width="100px" style="margin-right: 10px;" />
    <h1 style="font-size: 48px;">LaNavigation</h1>
    </div>
    <div id="inputState" style="display: none;" url="empty" objective="empty"></div>
    </div>
    </div>
    """

    title_history = """
    <div align="center">
    <h3>Steps</h3>
    </div>
    """

    js = """
function handleKeyboardEvents() {
    let isProcessingUrl = false;
    let isProcessingObjective = false;

    document.addEventListener('keydown', function(event) {
        // Handle Shift + Spacebar
        if (event.code === 'Space' && event.shiftKey) {
            event.preventDefault();

            if (isProcessingUrl) {
                console.log('Still processing previous URL action');
                return;
            }
            isProcessingUrl = true;

            const inputState = document.getElementById('inputState');
            const urlState = inputState.getAttribute('url') || 'empty';
            const urlInput = document.querySelector('#url-input');

            if (!urlInput) {
                console.log('URL input container not found');
                isProcessingUrl = false;
                return;
            }

            function findAndClickUrlButtons() {
                const urlButtonClose = urlInput.querySelectorAll('.svelte-rk35yg.padded');
                const urlButtons = urlInput.querySelectorAll('.record.record-button.svelte-1d9m1oy');

                if (urlState === 'empty') {
                    if (urlButtons.length > 0) {
                        console.log('Starting URL Input');
                        urlButtons[0].click();
                        inputState.setAttribute('url', 'in-use');
                        isProcessingUrl = false;
                    } else {
                        console.log('URL button not found for initial click');
                        setTimeout(findAndClickUrlButtons, 100);
                    }
                } else if (urlState === 'in-use') {
                    if (urlButtonClose.length > 0) {
                        console.log('Closing URL input');
                        urlButtonClose[0].click();
                        
                        setTimeout(() => {
                            const newUrlButtons = urlInput.querySelectorAll('.record.record-button.svelte-1d9m1oy');
                            if (newUrlButtons.length > 0) {
                                console.log('Restarting URL Input');
                                newUrlButtons[0].click();
                            } else {
                                console.log('URL button not found for restart');
                                inputState.setAttribute('url', 'empty');
                            }
                            isProcessingUrl = false;
                        }, 500);
                    } else {
                        console.log('Close button not found');
                        isProcessingUrl = false;
                    }
                }
            }

            findAndClickUrlButtons();
        } 
        // Handle Shift + B with similar logic
        else if (event.code === 'KeyB' && event.shiftKey) {
            event.preventDefault();

            if (isProcessingObjective) {
                console.log('Still processing previous objective action');
                return;
            }
            isProcessingObjective = true;

            const inputState = document.getElementById('inputState');
            const objectiveState = inputState.getAttribute('objective') || 'empty';
            const objectiveInput = document.querySelector('#objective-input');

            if (!objectiveInput) {
                console.log('Objective input container not found');
                isProcessingObjective = false;
                return;
            }

            function findAndClickObjectiveButtons() {
                const objectiveButtonClose = objectiveInput.querySelectorAll('.svelte-rk35yg.padded');
                const objectiveButtons = objectiveInput.querySelectorAll('.record.record-button.svelte-1d9m1oy');

                if (objectiveState === 'empty') {
                    if (objectiveButtons.length > 0) {
                        console.log('Starting Objective Input');
                        objectiveButtons[0].click();
                        inputState.setAttribute('objective', 'in-use');
                        isProcessingObjective = false;
                    } else {
                        console.log('Objective button not found for initial click');
                        setTimeout(findAndClickObjectiveButtons, 100);
                    }
                } else if (objectiveState === 'in-use') {
                    if (objectiveButtonClose.length > 0) {
                        console.log('Closing Objective input');
                        objectiveButtonClose[0].click();
                        
                        setTimeout(() => {
                            const newObjectiveButtons = objectiveInput.querySelectorAll('.record.record-button.svelte-1d9m1oy');
                            if (newObjectiveButtons.length > 0) {
                                console.log('Restarting Objective Input');
                                newObjectiveButtons[0].click();
                            } else {
                                console.log('Objective button not found for restart');
                                inputState.setAttribute('objective', 'empty');
                            }
                            isProcessingObjective = false;
                        }, 500);
                    } else {
                        console.log('Objective close button not found');
                        isProcessingObjective = false;
                    }
                }
            }

            findAndClickObjectiveButtons();
        } 
        // Handle Shift + N
        else if (event.code === 'KeyN' && event.shiftKey) {
            event.preventDefault();
            const buttons = document.querySelectorAll('.play-pause-button.icon.svelte-ije4bl');
            if (buttons.length > 0) {
                buttons[buttons.length - 1].click();
            } else {
                console.log('No play-pause buttons found.');
            }
        }
    });
}    """

    def __init__(
        self,
        objective: str,
        instructions: Optional[List[str]],
        agent: WebAgent,
        user_data=None,
        screenshot_ratio: float = 1,
    ):
        self.objective = objective
        self.instructions = instructions
        self.agent = agent
        self.user_data = user_data
        self.previous_val = None
        self.screenshot_ratio = screenshot_ratio

    def _init_driver(self):
        def init_driver_impl(url, img):
            prompt = "The following URL was collected with a microphone from a user. It might be noisy. If it seems like a valid url, repeat it back and don't say anything else. If not, provide the URL the user likely tried to navigate to, and nothing else. For example, if the URL is collected as 'https://www.Gotobbc.co.uk', the intended URL might have been 'https://www.bbc.co.uk'. Or if the URL was collected as 'https://www.visitwikipedia.org', the intended URL might have been 'https://www.wikipedia.org'. Either way, produce nothing but a precise URL. '" + url + "'"
            verified_url_response = self.verifier.complete(prompt)
            verified_url = verified_url_response.text
            self.agent.get(verified_url)
            

            ret = self.agent.action_engine.driver.get_screenshot_as_png()
            ret = BytesIO(ret)
            ret = Image.open(ret)
            img = ret
            return verified_url, img

        return init_driver_impl
    
    def _play_audio(self):
        def play_audio(chatbot):
            print(chatbot)
            filename = generate_and_play_tts(chatbot[-1]["content"])
            audio_file = gr.Audio(value=filename, autoplay=True, label="Play Audio File", visible=True)
            return audio_file

        return play_audio

    def _process_instructions(self):
        def process_instructions_impl(objective, url_input, image_display, history):
            msg = gr.ChatMessage(
                role="assistant", content="⏳ Thinking of next steps..."
            )
            history.append(msg)
            yield objective, url_input, image_display, history
            self.agent.action_engine.set_gradio_mode_all(
                True, objective, url_input, image_display, history, 
            )
            self.agent.clean_screenshot_folder = False

            yield from self.agent._run_demo(
                objective,
                self.user_data,
                False,
                objective,
                url_input,
                image_display,
                history,
                self.screenshot_ratio,
            )

            return objective, url_input, image_display, history

        return process_instructions_impl

    def __add_message(self):
        def add_message(history, message):
            history.clear()
            return history

        return add_message

    def refresh_img_dislay(self, url, image_display):
        img = self.agent.driver.get_screenshot_as_png()
        img = BytesIO(img)
        img = Image.open(img)
        if self.screenshot_ratio != 1:
            img = img.resize(
                (
                    int(img.width / self.screenshot_ratio),
                    int(img.height / self.screenshot_ratio),
                )
            )
        image_display = img
        return url, image_display

    def launch(self, server_port=7860, share=True, debug=True):
        with gr.Blocks(
            gr.themes.Default(primary_hue="blue", secondary_hue="neutral"), css=self.css, js=self.js
        ) as demo:
            with gr.Tab(""):
                with gr.Row():
                    gr.HTML(self.title)

                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            audio_input = gr.Audio(sources="microphone", elem_id="url-input")
                        with gr.Row():
                            transcription_output = gr.Textbox(
                                value=self.agent.action_engine.driver.get_url(),
                                scale=9,
                                type="text",
                                placeholder="The transcribed URL will appear here...",
                                label="URL",
                                visible=True,
                                max_lines=1,
                            )
                        with gr.Row():
                            audio_input_objective = gr.Audio(sources="microphone", elem_id="objective-input")
                        with gr.Row():
                            transcription_output_objective = gr.Textbox(
                                value=self.objective,
                                scale=9,
                                type="text",
                                placeholder="The transcribed objective will appear here...",
                                label="Objective",
                                visible=True,
                                max_lines=1,
                            )

                        audio_file = gr.Audio(label="Play Audio File", visible=True, autoplay=True)
                with gr.Row(variant="panel", equal_height=True):
                    with gr.Column(scale=8):
                        image_display = gr.Image(
                            label="Browser", interactive=False, height="100%"
                        )
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            [],
                            label="Agent output",
                            elem_id="history",
                            type="messages",
                            bubble_full_width=False,
                            height="100%",
                            placeholder="Agent output will be shown here\n",
                            layout="bubble",
                        )

                # Automatically submit the transcription after it's received
                audio_input.change(fn=transcribe, inputs=audio_input, outputs=transcription_output).then(
                    self._init_driver(),
                    inputs=[transcription_output, image_display],
                    outputs=[transcription_output, image_display],
                )
                
                audio_input_objective.change(
                    fn=transcribe_objective,
                    inputs=audio_input_objective,
                    outputs=transcription_output_objective,
                ).then(
                    self.__add_message(),
                    inputs=[chatbot, transcription_output_objective],
                    outputs=[chatbot],
                ).then(
                    self._process_instructions(),
                    inputs=[
                        transcription_output_objective,
                        transcription_output,
                        image_display,
                        chatbot,
                    ],
                    outputs=[
                        transcription_output_objective,
                        transcription_output,
                        image_display,
                        chatbot,
                    ],
                ).then(
                    self._play_audio(),
                    inputs=[chatbot],
                    outputs=[audio_file],
                )

                if self.agent.driver.get_url() is not None:
                    demo.load(
                        fn=self.refresh_img_dislay,
                        inputs=[transcription_output, image_display],
                        outputs=[transcription_output, image_display],
                        show_progress=False,
                    )
        demo.launch(server_port=server_port, share=True, debug=True, allowed_paths=["announcing-mistral.png"])



mistral_api_key = load_key("mistral_key", file='Neel_config.yaml')
# google_api_key = load_key_Titas("google_key", file='Titas_config.yaml')
google_api_key = load_key("google_key", file='Neel_config.yaml')

selenium_driver = SeleniumDriver()
llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key, temperature=0.01)
# llm = Gemini(model_name="models/gemini-1.5-flash-latest", temperature=0.01)

mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-flash-latest", api_key=google_api_key, temperature=0.01, )
# pixtral = PixtralWrapper()
os.environ[
    "MISTRAL_API_KEY"
] = mistral_api_key
pixtral = MistralAIMultiModal(
    model="pixtral-12b-2409", max_new_tokens=300
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

context = Context(llm, mm_llm=mm_llm, embedding=embed_model)

python_engine = PythonEngine(
    driver=selenium_driver,
    llm=llm,
    embedding=embed_model,
    ocr_mm_llm=mm_llm
    )

# Initialize a WorldModel and ActionEngine passing them your models
world_model = WorldModel(mm_llm=mm_llm)
action_engine = ActionEngine(driver=selenium_driver, llm=llm, embedding=embed_model, python_engine=python_engine)
# context.action_engine = action_engine
# context.driver = selenium_driver

# Build agent & run query
agent = WebAgent(world_model, action_engine)

grad = GradioAgentDemo("", "", agent)
grad.verifier = pixtral
grad.launch(server_port=8502, share=True, debug=True)
