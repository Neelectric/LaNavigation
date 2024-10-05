### Main driver file for explanations
###
###

# System imports
import time

# External imports
import torch

# Local imports

# explanation function
def explain(url):
    explanation = ""
    try:
        screenshot_location = take_screenshot(url)
    except:
        print("Taking a screenshot seems to have failed")
    explanation = prompt_pixtral(screenshot_location)
    
    return explanation