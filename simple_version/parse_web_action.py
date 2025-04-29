import logging
import asyncio
from rapidfuzz import process
import numpy as np


async def _execute_action(page, action: dict) -> str:
    """
    Execute a single browser action using Playwright and return observation
    """
    logging.info(f"Executing action: {action}")
    choices = ["navigate", "enter_text", "enter_search", "press_enter", "extract_text", "click", "identify_information", "highlight"]
    query = action["action"]
    best_match = process.extractOne(query, choices)[0]
    logging.info(f"Found best matched action: {best_match}")
    try:
        observation = ""
        if best_match == "navigate":
            url = list(action["parameters"].values())[0]
            await page.goto(url, wait_until="networkidle", timeout=10000)
            observation = f"Navigated to {url}"
        elif best_match in ["enter_text", "enter_search"]:
            selector = 'input[name="q"], textarea[name="q"]'
            element = await page.wait_for_selector(selector, timeout=5000)
            key = list(action["parameters"].keys())[0]
            await element.fill(action["parameters"][key])
            observation = f"""Typed '{action["parameters"][key]}'"""
            
        elif best_match == "press_enter":
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("networkidle", timeout=10000)
            observation = f"Pressed Enter"
            
        elif best_match in ["extract_text", "identify_information", "highlight"]:
            await page.wait_for_load_state("networkidle", timeout=10000)
            h3 = page.locator('h3').first
            firstResult = await h3.inner_text()
            observation = f"Retrieved " + firstResult
        
        elif best_match == "click":
            button_name = list(action["parameters"].values())[0]
            button_name = button_name.replace(' button', '')
            num = await page.get_by_role("button").count()
            if num > 0:
                buttons = page.get_by_role('button')
                texts = []
                for i in range(num):
                    texts.append(await buttons.nth(i).get_attribute("value"))
                best_button = process.extractOne(button_name, texts)[0]
                logging.info("best matched button: ", best_button)
                ind = np.where([x == best_button for x in texts])[0][0]
                await buttons.nth(ind).click()
            else:
                logging.error("No button is found on the page.")
            observation = f"Clicked {button_name}"
        await asyncio.sleep(1)
        return observation
    except Exception as e:
        error_msg = f"Error executing {action}: {str(e)}"
        logging.error(error_msg)
        return error_msg