### Main driver file for explanations
###
###

# System imports
import time
import asyncio

# External imports
import torch

# Local imports
from src.screenshot import capture_website_screenshot
from src.pixtral_prompting import prompt_pixtral


# explanation function
def explain(url):
    explanation = ""
    try:
        # screenshot_location = capture_website_screenshot(url)
        screenshot_location = 'screenshots/screenshot.png'
        capture_website_screenshot(url, output_file=screenshot_location)
    except:
        print("Taking a screenshot seems to have failed")
    screenshot_location = "./" + screenshot_location
    explanation = prompt_pixtral(screenshot_location)

    return explanation


# Example usage:
if __name__ == "__main__":
    website_url = "https://en.wikipedia.org/wiki/Donald_Trump"
    explain(website_url)