
from lavague.core import WorldModel, ActionEngine, PythonEngine
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from typing import Optional, List, Mapping, Any
from llama_index.llms.mistralai import MistralAI

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.pixtral_prompting import prompt_pixtral_text


class OurLLM(CustomLLM):
    context_window: int = 128000
    num_output: int = 8192 * 4
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        for i in range(5):
            output = prompt_pixtral_text(prompt)
            if output is not None:
                break
        print(output)
        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)


# define our LLM
mm_llm = OurLLM()
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# Initialize the Selenium driver
selenium_driver = SeleniumDriver()
python_engine = PythonEngine(
    driver=selenium_driver,
    llm=mm_llm,
    embedding=embed_model,
    ocr_mm_llm=mm_llm
    )

# Initialize a WorldModel and ActionEngine passing them your models
world_model = WorldModel(mm_llm=mm_llm)
action_engine = ActionEngine(driver=selenium_driver, llm=mm_llm, embedding=embed_model, python_engine=python_engine)

# Create your agent
agent = WebAgent(world_model, action_engine)

# agent.get("https://huggingface.co/docs")
# agent.run("Go to the quicktour page of PEFT. Then provide a summary of the page.")
agent.get("https://forms.office.com/r/prD2rCGp6i")
agent.run("Describe concretely the main elements of this page")