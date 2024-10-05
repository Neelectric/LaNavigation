import asyncio
from pyppeteer import launch

async def take_screenshot(url, output_file):
    # Launch headless browser
    browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
    
    # Open a new page
    page = await browser.newPage()
    
    # Go to the specified URL
    await page.goto(url)
    
    # Take a screenshot
    await page.screenshot({'path': output_file})
    
    # Close the browser
    await browser.close()

def capture_website_screenshot(url, output_file='screenshot.png'):
    asyncio.get_event_loop().run_until_complete(take_screenshot(url, output_file))

# Example usage:
if __name__ == "__main__":
    website_url = "https://www.vrk.lt/en/home"
    output_path = "website_screenshot.png"
    capture_website_screenshot(website_url, output_path)
    print(f"Screenshot saved as {output_path}")
