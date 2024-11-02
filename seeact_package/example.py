import argparse
import asyncio
import os
import json
import toml
from seeact.agent import SeeActAgent
from dotenv import load_dotenv

load_dotenv()
# Setup your API Key here, or pass through environment
# openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["GEMINI_API_KEY"] = "Your API KEY Here"

async def run_agent():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config_path", help="Path to the TOML configuration file.", type=str, metavar='config',
                        default=f"{os.path.join('config', 'example_config.toml')}")
    args = parser.parse_args()
    base_dir = os.getcwd()
    config_path = args.config_path if args.config_path else None
    agent = SeeActAgent(config_path=config_path)
    await agent.start()
    while not agent.complete_flag:
        prediction_dict = await agent.predict()
        await agent.execute(prediction_dict)
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(run_agent())
