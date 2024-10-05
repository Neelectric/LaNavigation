import httpx

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
data = {
    "model": "mistralai/Pixtral-12B-2409",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in a short sentence."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://picsum.photos/id/237/200/300"},
                },
            ],
        }
    ],
}

response = httpx.post(url, headers=headers, json=data)

print(response.json())

def prompt_pixtral(url):
    explanation = ""
    return explanation