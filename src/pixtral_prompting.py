### Any interactions with pixtral are contained in this file
###
###

# System imports
import httpx
import base64
import requests
import os

# External imports
from mistralai import Mistral

# Local imports

# taken directly from https://docs.mistral.ai/capabilities/vision/, allows us to encode an image to base64 given path
def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    
def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"

def prompt_pixtral_text(prompt):
    pixtral_url = "http://localhost:8001/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
    data = {
        "model": "mistralai/Pixtral-12B-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    response = httpx.post(pixtral_url, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]

    # print(response.json())
    # print(content)
    return content


def pixtral_explain_screenshot(screenshot_location):
    pixtral_url = "http://localhost:8001/v1/chat/completions"
    # base64_image = encode_image(screenshot_location)
    image_source = file_to_data_url(screenshot_location)

    headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
    data = {
        "model": "mistralai/Pixtral-12B-2409",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that guides visually impaired people through websites. The following is a screenshot taken from a website that a user has navigated to. Please first provide a brief summary of the whole page. Then, generate a detailed description of all of the content on the website, taking special care to provide all information that a visually impaired person might be interested in. Skip any content that may be unnecessary or irrelevant."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_source} ,
                    },
                ],
            }
        ],
    }

    response = httpx.post(pixtral_url, headers=headers, json=data)

    # print(response.json())
    print(response.json()["choices"][0]["message"]["content"])
    return response.json()