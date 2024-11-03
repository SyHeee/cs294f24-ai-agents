import argparse
import asyncio
import os
import json
import toml
from seeact.agent import SeeActAgent
from seeact.mcts_agent import MctsSeeActAgent
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
    parser.add_argument("-s", "--search_method", help="Set the action search method, choose from 'basic' and 'mcts'", type=str, metavar='[basic | mcts]',
                        default='basic')
    args = parser.parse_args()
    base_dir = os.getcwd()
    config_path = args.config_path if args.config_path else None
    method = args.search_method if args.search_method else 'basic'
    if method == "mcts":
        agent = MctsSeeActAgent(config_path=config_path)
        try:
            await agent.start()
            await agent.initialize_mcts()
            max_iterations = 5
            for i in range(max_iterations):
                print(f"\nIteration {i + 1}/{max_iterations}")
                
                # Get prediction using MCTS
                prediction = await agent.mcts_predict(num_simulations=10)
                
                if prediction is None:
                    print("No valid prediction found")
                    break
                # Execute the predicted action
                result = await agent.execute(prediction)
                if agent.complete_flag:
                    print("Task completed successfully!")
                    break
                    
                print(f"Action taken: {prediction['action']}")
                print(f"Result: {'Success' if result == 0 else 'Failure'}")
        except Exception as e:
            print(f"Error during test: {e}")
        finally:
            await agent.cleanup_mcts()
            await agent.stop()
    else:
        agent = SeeActAgent(config_path=config_path)
        await agent.start()
        while not agent.complete_flag:
            prediction_dict = await agent.predict()
            await agent.execute(prediction_dict)
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(run_agent())
