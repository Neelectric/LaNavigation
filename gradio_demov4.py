from io import BytesIO
import queue
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
from src.load_config import load_key
from src.pixtral_wrapper import PixtralWrapper
from lavague.drivers.selenium import SeleniumDriver
from lavague.core import WorldModel, ActionEngine, PythonEngine

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=0)

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
    <img src="https://raw.githubusercontent.com/lavague-ai/LaVague/new_gradio/docs/assets/logo.png" width=40px/>
    </div>
    <div class='child' style="display: inline-block !important;">
    <h1>LaVague</h1>
    </div>
    </div>
    """

    title_history = """
    <div align="center">
    <h3>Steps</h3>
    </div>
    """

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
            print(url)
            self.agent.get(url)

            ret = self.agent.action_engine.driver.get_screenshot_as_png()
            ret = BytesIO(ret)
            ret = Image.open(ret)
            img = ret
            return url, img

        return init_driver_impl

    def _process_instructions(self):
        def process_instructions_impl(objective, url_input, image_display, history):
            msg = gr.ChatMessage(
                role="assistant", content="⏳ Thinking of next steps..."
            )
            history.append(msg)
            yield objective, url_input, image_display, history
            self.agent.action_engine.set_gradio_mode_all(
                True, objective, url_input, image_display, history
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
            gr.themes.Default(primary_hue="blue", secondary_hue="neutral"), css=self.css
        ) as demo:
            with gr.Tab(""):
                with gr.Row():
                    gr.HTML(self.title)

                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            audio_input = gr.Audio(sources="microphone", label="Record Audio")
                            transcription_output = gr.Textbox(
                                value=self.agent.action_engine.driver.get_url(),
                                scale=9,
                                type="text",
                                label="The transcribed URL will appear here...",
                                visible=True,
                                max_lines=1,
                            )
                        with gr.Row():
                            audio_input_objective = gr.Audio(sources="microphone", label="Record Audio Objective")
                            transcription_output_objective = gr.Textbox(
                                value=self.objective,
                                scale=9,
                                type="text",
                                label="The transcribed objective will appear here...",
                                visible=True,
                                max_lines=1,
                            )
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
                # Use the image_updater generator function
                # submission handling
                # audio_input.change(fn=transcribe, inputs=audio_input, outputs=transcription_output).then(
                #     self._init_driver(),
                #     inputs=[audio_input, transcription_output],
                #     outputs=[audio_input],
                # )
                # objective submission handling
                # Automatically submit the transcription after it's received
                audio_input.change(fn=transcribe, inputs=audio_input, outputs=transcription_output).then(
                    self._init_driver(),
                    inputs=[transcription_output],
                    outputs=[transcription_output],
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
                )
                if self.agent.driver.get_url() is not None:
                    demo.load(
                        fn=self.refresh_img_dislay,
                        inputs=[transcription_output, image_display],
                        outputs=[transcription_output, image_display],
                        show_progress=False,
                    )
        demo.launch(server_port=server_port, share=True, debug=True)



mistral_api_key = load_key("mistral_key", file='Neel_config.yaml')
google_api_key = load_key("google_key", file='Neel_config.yaml')

selenium_driver = SeleniumDriver()
llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key, temperature=0.01)
# llm = Gemini(model_name="models/gemini-1.5-flash-latest", temperature=0.01)
mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-flash-latest", api_key=google_api_key, temperature=0.01)
# llm = PixtralWrapper()
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
context.action_engine = action_engine
context.driver = selenium_driver

# Build agent & run query
agent = WebAgent(world_model, action_engine)

grad = GradioAgentDemo("", selenium_driver, context)
grad.launch(server_port=7891, share=True, debug=True)
