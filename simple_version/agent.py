import asyncio
from playwright.async_api import async_playwright, TimeoutError
import os
from typing import List, Dict, Any
import anthropic
from llmodels import gpt, gpt4
from functools import partial
import logging
from datetime import datetime
import time

from prompts import _initialize_prompt
from utils import setup_logging, get_log_dir

class Agent:
    def __init__(self, llm_type: str = "gpt-3.5-turbo-16k", temperature: float = 0.7):
        self.llm_type = llm_type
        self.temperature = temperature
        self.browser = None
        self.page = None
        self.context = None
        self.trajectory: List[Dict[str, str]] = []
        self.step_counter = 0
        
        self.log_dir, self.screenshot_dir = get_log_dir()
        self.logger = setup_logging(self.log_dir, 'Agent')

    async def setup(self):
        self.logger.info("Setting up the browser...")
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        self.logger.info("Browser setup complete.")

    async def cleanup(self):
        self.logger.info("Cleaning up...")
        await self.browser.close()
        self.logger.info("Browser closed.")

    async def take_screenshot(self):
        self.step_counter += 1
        screenshot_path = os.path.join(self.screenshot_dir, f"step_{self.step_counter}.png")
        await self.page.screenshot(path=screenshot_path)
        self.logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path

    async def log_action(self, action: str, observation: str):
        screenshot_path = await self.take_screenshot()
        log_entry = {
            "step": self.step_counter,
            "action": action,
            "observation": observation,
            "screenshot": screenshot_path
        }
        self.trajectory.append(log_entry)
        self.logger.info(f"Step {self.step_counter}: {action}")
        self.logger.info(f"Observation: {observation}")

    async def search_web(self, query: str, url="https://www.google.com"):
        try:
            self.logger.info(f"Navigating to Google...")
            await self.page.goto(url, wait_until="networkidle", timeout=10000)
            await self.log_action("Navigated to Google", "Google homepage loaded")
            
            self.logger.info(f"Searching for: {query}") 
            # TODO: add the fill text and search in llm function calling 
            search_input = await self.page.wait_for_selector('input[name="q"]', state="visible", timeout=5000)
            await search_input.type(query, delay=50)  # Reduced delay
            await asyncio.sleep(0.5)  # Reduced sleep time
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_load_state("networkidle", timeout=10000)
            
            title = await self.page.title()
            await self.log_action(f"Searched Google for: {query}", f"Search results page: {title}")
        
        except TimeoutError as e:
            error_msg = f"Timeout error occurred: {e}"
            self.logger.error(error_msg)
            await self.log_action(f"Failed to search Google for: {query}", error_msg)
        
        except Exception as e:
            error_msg = f"An error occurred during Google search: {e}"
            self.logger.error(error_msg)
            await self.log_action(f"Failed to search Google for: {query}", error_msg)

    async def get_llm_response(self, prompt: str) -> str:
        self.logger.info("Getting LLM response...")
        start_time = time.time()
        try:
            if self.llm_type == "gpt-4o":
                import pdb; pdb.set_trace() 
                response = await asyncio.wait_for(
                    asyncio.to_thread(gpt4, prompt, model=self.llm_type, temperature=self.temperature, max_tokens=15000),
                    timeout=30  # 30 second timeout
                )            
            elif self.llm_type == "claude":
                client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.completion,
                        prompt=prompt,
                        model="claude-2",
                        max_tokens_to_sample=100,
                        temperature=self.temperature
                    ),
                    timeout=30  # 30 second timeout
                )
                response = [response.completion]
            else:
                raise ValueError(f"Unsupported LLM type: {self.llm_type}")
            
            llm_response = response[0].strip()
            self.logger.info(f"LLM Response received in {time.time() - start_time:.2f} seconds")
            self.logger.debug(f"LLM Response: {llm_response}")
            return llm_response
        except asyncio.TimeoutError:
            self.logger.error(f"LLM response timed out after 30 seconds")
            return "ERROR: LLM response timed out"
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return f"ERROR: {str(e)}"

    async def run(self, initial_query: str, max_steps: int = 5):
        await self.setup()
        try:
            await self.search_web(initial_query)
            
            for step in range(max_steps):
                self.logger.info(f"Starting step {step + 1} of {max_steps}")
                if step >= 0:
                    prompt = self.generate_prompt()
                # else:
                #     prompt = _initialize_prompt()
                next_action = await self.get_llm_response(prompt)
                
                if next_action.startswith("ERROR:"):
                    self.logger.error(f"LLM error: {next_action}")
                    await self.log_action("LLM Error", next_action)
                    break
                
                if next_action.lower().startswith("click"):
                    element_to_click = next_action.split("click", 1)[1].strip()
                    try:
                        await self.page.click(f"text={element_to_click}", timeout=5000)
                        await self.page.wait_for_load_state("networkidle", timeout=10000)
                        await self.log_action(f"Clicked on: {element_to_click}", await self.page.title())
                    except Exception as e:
                        await self.log_action(f"Failed to click on: {element_to_click}", str(e))
                
                elif next_action.lower().startswith("search"):
                    query = next_action.split("search", 1)[1].strip()
                    await self.search_web(query)
                
                else:
                    await self.log_action(f"Unsupported action: {next_action}", "Skipping this step")
        
        finally:
            await self.cleanup()
        
        return self.trajectory

    def generate_prompt(self) -> str:
        prompt = "Based on the following trajectory, what should be the next action?\n\n"
        for step in self.trajectory[-5:]:  # Only use the last 5 steps to keep the prompt manageable
            prompt += f"Step {step['step']}:\nAction: {step['action']}\nObservation: {step['observation']}\n\n"
        prompt += "Next Action:"
        return prompt

async def main():
    agent = Agent(llm_type="gpt-4o")
    query = "What is the capital of France?"
    trajectory = await agent.run(query)
    
    print("\nFinal Trajectory:")
    for step in trajectory:
        print(f"Step {step['step']}:")
        print(f"Action: {step['action']}")
        print(f"Observation: {step['observation']}")
        print(f"Screenshot: {step['screenshot']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
