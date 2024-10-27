import asyncio
from playwright.async_api import async_playwright, TimeoutError
import os
from typing import List, Dict, Any, Optional
import anthropic
from llmodels import llm, MAX_TOKENS
from functools import partial
import logging
from datetime import datetime
import time
import json

from prompts import _initialize_prompt
from utils import setup_logging, get_log_dir
from parse_web_action import execute_action

class Agent:
    def __init__(self, 
                 llm_type: str = "gpt-4o-mini", 
                 temperature: float = 0.3,
                 headless: bool = False):
        self.llm_type = llm_type
        self.temperature = temperature
        self.headless = headless
        self.browser = None
        self.page = None
        self.context = None
        self.trajectory: List[Dict[str, str]] = []
        self.step_counter = 0

        self.prompts = _initialize_prompt()
        
        self.log_dir, self.screenshot_dir = get_log_dir()
        self.logger = setup_logging(self.log_dir, 'Agent')

    async def analyze_task(self, query: str) -> Dict:
        """
        Analyze user query using LLM to generate web action plan
        """
        system_prompt = """Analyze the user's query and determine the most appropriate web actions.
        Generate a specific plan including website to visit, search terms, and target elements."""
        
        user_prompt = f"""Query: {query}
        Generate a structured web action plan in json."""
        
        combined_prompt = f"{system_prompt}\n{user_prompt}"
        
        try:
            ### 
            llm_response = await self.get_llm_response(combined_prompt)
            # Parse LLM response into action plan
            action_plan = {
                "website": "https://www.google.com",  # Default fallback
                "search_query": query,
                "target_elements": ["search input", "search results"],
                "actions": []
            }
            
            try:
                # Try to parse LLM response as JSON if it's formatted that way
                parsed_response = json.loads(llm_response)
                action_plan.update(parsed_response)
            except json.JSONDecodeError:
                # If not JSON, use the raw response as the search query
                action_plan["search_query"] = llm_response.strip()
            
            self.logger.info(f"Generated action plan for query: {query}")
            return action_plan
            
        except Exception as e:
            self.logger.error(f"Error in task analysis: {e}")
            return {
                "website": "https://www.google.com",
                "search_query": query,
                "target_elements": ["search input", "search results"],
                "actions": []
            }

    async def ground_action(self, action_plan: Dict) -> List[Dict]:
        """
        Convert high-level action plan into specific browser actions
        """
        self.logger.info("Grounding actions from plan...")
        
        try:
            # Default set of actions based on the plan
            actions = []
            
            # Add actions from the plan
            for action in action_plan.get("actions", []):
                actions.append(action)
            
            self.logger.info(f"Grounded {len(actions)} actions")
            return actions
            
        except Exception as e:
            self.logger.error(f"Error in action grounding: {e}")
            return []

    async def execute_action(self, action: Dict) -> str:
        """
        Execute a single browser action using Playwright and return observation
        """
        observation = await execute_action(self.page, action)
        await self.log_action(action["action"], observation)
        return observation

    async def setup(self):
        """Initialize browser and create new page"""
        self.logger.info("Setting up the browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        self.logger.info("Browser setup complete.")

    async def cleanup(self):
        """Clean up browser resources"""
        self.logger.info("Cleaning up...")
        try:
            if self.page:
                await self.page.close()  # Close the page if it exists
                self.page = None  # Clear reference
            if self.context:
                await self.context.close()  # Close the context if it exists
                self.context = None  # Clear reference
            if self.browser:
                await self.browser.close()  # Close the browser if it exists
                self.browser = None  # Clear reference
        finally:
            if self.playwright:
                await self.playwright.stop()  # Stop the Playwright instance
                self.playwright = None  # Clear reference
        self.logger.info("Browser closed.")

    async def take_screenshot(self):
        """Take and save a screenshot"""
        self.step_counter += 1
        screenshot_path = os.path.join(self.screenshot_dir, f"step_{self.step_counter}.png")
        await self.page.screenshot(path=screenshot_path)
        self.logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path

    async def log_action(self, action: str, observation: str):
        """Log action and observation with screenshot"""
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
    
    async def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM with timeout handling"""
        self.logger.info("Getting LLM response...")
        start_time = time.time()
        
        try:            
            response = await asyncio.wait_for(
                llm(
                    prompt,
                    model=self.llm_type,
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS
                ),
                timeout=30
            )
            
            llm_response = response[0].strip()
            self.logger.info(f"LLM Response received in {time.time() - start_time:.2f} seconds")
            return llm_response
            
        except asyncio.TimeoutError:
            self.logger.error("LLM response timed out after 30 seconds")
            return "ERROR: LLM response timed out"
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return f"ERROR: {str(e)}"
        
    async def is_done(self, query: str) -> dict:
        """
        Analyze the trajectory to see if one should stop
        """
        system_prompt = """Analyze the query and the existing search results.
        Return whether one can answer the question given the search results."""
        
        user_prompt = f"""Query: {query}, Current search results: {self.trajectory}
        Can we stop searching and give answer for the query right now? 
        Answer in json, must contain attribute can_stop_search (True or False), 
        reason (Why we can stop or not), 
        final_answer (final answer to the query if we can answer the query)."""
        
        combined_prompt = f"{system_prompt}\n{user_prompt}"

        llm_response = await self.get_llm_response(combined_prompt)

        self.logger.info(f"got response : {llm_response}")
        return json.loads(llm_response)
            
    async def run(self, initial_query: str, max_steps: int = 5):
        """Main execution loop"""
        try:
            await self.setup()
            # Analyze the task first
            action_plan = await self.analyze_task(initial_query)
            grounded_actions = await self.ground_action(action_plan)
            # Execute initial actions
            for action in grounded_actions:
                await self.execute_action(action)
            
            result = await self.is_done(initial_query)

            if result["can_stop_search"]:
                self.trajectory.append(result)
            else:
                # Continue with dynamic action generation
                for step in range(len(grounded_actions), max_steps):
                    self.logger.info(f"Starting step {step + 1} of {max_steps}")
                    prompt = self.generate_prompt()
                    next_action = await self.get_llm_response(prompt)
                    
                    if next_action.startswith("ERROR:"):
                        self.logger.error(f"LLM error: {next_action}")
                        break
                    
                    # Convert LLM response to action
                    try:
                        action = {
                            "action": "click" if next_action.lower().startswith("click") else "search",
                            "value": next_action.split(None, 1)[1].strip(),
                            "description": next_action
                        }
                        await self.execute_action(action)
                    except Exception as e:
                        self.logger.error(f"Error executing action: {e}")
                        break
        
        finally:
            await self.cleanup()
        
        return self.trajectory

    def generate_prompt(self) -> str:
        """Generate prompt for next action based on trajectory"""
        prompt = "Based on the following trajectory, what should be the next action in json?\n\n"
        for step in self.trajectory[-5:]:  # Only use the last 5 steps
            prompt += f"Step {step['step']}:\nAction: {step['action']}\nObservation: {step['observation']}\n\n"
        prompt += "Next Action:"
        return prompt

async def main():
    agent = Agent(llm_type="gpt-4o-mini")
    query = "What is the capital city of France?"
    trajectory = await agent.run(query)
    
    print("\nFinal Trajectory:")
    for step in trajectory[:-1]:
        print(f"Step {step['step']}:")
        print(f"Action: {step['action']}")
        print(f"Observation: {step['observation']}")
        print(f"Screenshot: {step['screenshot']}")
        print()
    print(trajectory[-1]["final_answer"])
    

if __name__ == "__main__":
    asyncio.run(main())