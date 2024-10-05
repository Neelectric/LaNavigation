# Local pixtral wrapper that works with llama_index
# System imports
from typing import Optional, List, Mapping, Any

# External imports
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

# Local imports
from src.pixtral_prompting import prompt_pixtral_text, prompt_pixtral_text_and_image



class PixtralWrapper(CustomLLM):
    context_window: int = 128000
    num_output: int = 8192 * 8
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
        print("calling pixtral")
        image_documents = kwargs.get("image_documents", [])
        file_paths = []
        if len(image_documents) > 1:
            raise ValueError("PixtralWrapper currently only supports one image at a time")
        if len(image_documents) == 1:
            for elt in image_documents:
                elt = elt.to_dict()
                file_paths.append(elt["metadata"]["file_path"])
            screenshot_location = file_paths[0]
            output = prompt_pixtral_text_and_image(prompt, screenshot_location)
            return CompletionResponse(text=output)
        else:
            for i in range(5):
                try:
                    # print(prompt)
                    output = prompt_pixtral_text(prompt)
                except:
                    output = None
                if output is not None:
                    break
            return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)