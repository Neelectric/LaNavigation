from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"
max_img_per_msg = 3

sampling_params = SamplingParams(max_tokens=8192)
llm = LLM(
    model=model_name,
    tokenizer_mode="mistral",
    load_format="mistral",
    config_format="mistral",
    limit_mm_per_prompt={"image": max_img_per_msg},
)

urls = [f"https://picsum.photos/id/{id}/512/512" for id in ["1", "11", "111"]]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            ] + [{"type": "image_url", "image_url": {"url": f"{u}"}} for u in urls],
    },
]

res = llm.chat(messages=messages, sampling_params=sampling_params)
print(res[0].outputs[0].text)