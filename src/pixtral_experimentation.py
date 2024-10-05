
# System imports
import time

# External imports
from lavague.core import WorldModel, ActionEngine, PythonEngine
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from llama_index.llms.mistralai import MistralAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from load_config import load_key

time_before = time.time()

mistral_api_key = load_key("mistral_key", file='Neel_config.yaml')

# define our LLMs
llm = MistralAI(model="mistral-large-latest", api_key=mistral_api_key)
resp = llm.complete("Paul Graham is ")
print(resp)
time_after = time.time()
print("Time taken: ", time_after - time_before)