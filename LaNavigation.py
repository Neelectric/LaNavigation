
import os

import google.generativeai as genai
from lavague.core import WorldModel, ActionEngine, PythonEngine
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.load_config import load_key
from src.pixtral_wrapper import PixtralWrapper


mistral_api_key = load_key("mistral_key", file='Neel_config.yaml')
google_api_key = load_key("google_key", file='Neel_config.yaml')
# genai.configure(api_key=google_api_key)

os.environ["GOOGLE_API_KEY"] = google_api_key

# define our LLMs
llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key, temperature=0.01)
# llm = Gemini(model_name="models/gemini-1.5-flash-latest", temperature=0.01)
mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-flash-latest", temperature=0.01)
# llm = PixtralWrapper()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
# embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=mistral_api_key)


# Initialize the Selenium driver
selenium_driver = SeleniumDriver()
python_engine = PythonEngine(
    driver=selenium_driver,
    llm=llm,
    embedding=embed_model,
    ocr_mm_llm=mm_llm
    )

# Initialize a WorldModel and ActionEngine passing them your models
world_model = WorldModel(mm_llm=mm_llm)
action_engine = ActionEngine(driver=selenium_driver, llm=llm, embedding=embed_model, python_engine=python_engine)

# Create your agent
agent = WebAgent(world_model, action_engine)

# agent.get("https://huggingface.co/docs")
# agent.run("Go to the quicktour page of PEFT. Then provide a summary of the page.")
data = """
- name: John Doe
- email: john.doe@gmail.com
- number of guests: 3
- allergies: peanuts
- activites: Charades
"""
agent.get("https://forms.office.com/r/prD2rCGp6i")
# agent.run("Click on the Start Now Button, and then report what the form is about. Do not submit any forms."
#         #   user_data=data
#           )
# agent.run_step(objective)
agent.run("Fill out this form. Keep in mind the following data: " + data)