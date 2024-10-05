
from lavague.core import WorldModel, ActionEngine, PythonEngine
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from llama_index.llms.mistralai import MistralAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.load_config import load_key
from src.pixtral_wrapper import PixtralWrapper


mistral_api_key = load_key("mistral_key", file='Neel_config.yaml')

# define our LLMs
llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key)
mm_llm = PixtralWrapper()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


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
agent.get("https://forms.office.com/r/prD2rCGp6i")
agent.run("Describe concretely the main elements of this page")