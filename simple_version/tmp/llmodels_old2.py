import os
import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
from dotenv import load_dotenv

import openai
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
import anthropic
import google.generativeai as genai
from transformers import GPT2Tokenizer

from utils import setup_logging, get_log_dir
log_dir, _ = get_log_dir()
logger = setup_logging(log_dir, 'LLM')


# Load environment variables from .env file
load_dotenv()

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 15000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

class APIKeyManager:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.openai_key:
            logger.warning("OPENAI_API_KEY is not set in .env file")
        if not self.anthropic_key:
            logger.warning("ANTHROPIC_API_KEY is not set in .env file")
        if not self.google_key:
            logger.warning("GOOGLE_API_KEY is not set in .env file")
        else:
            genai.configure(api_key=self.google_key)
        
        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=self.openai_key)

api_keys = APIKeyManager()

def retry_with_exponential_backoff(
    func: Callable,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (APIError, APIConnectionError, RateLimitError),
):
    """Retry an async function with exponential backoff."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        
        while True:
            try:
                return await func(*args, **kwargs)
                
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warning(f"Attempt {num_retries} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                
            except Exception as e:
                raise e
                
    return wrapper

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def claude(prompt, model="claude-2", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    client = anthropic.Client(api_key=anthropic_api_key)
    outputs = []
    for _ in range(n):
        response = client.completion(
            prompt=prompt,
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            stop_sequences=stop
        )
        outputs.append(response.completion.strip())
    return outputs

def gemini(prompt, model="gemini-pro", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    model = genai.GenerativeModel(model_name=model)
    outputs = []
    for _ in range(n):
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=stop
            )
        )
        outputs.append(response.text.strip())
    return outputs

async def async_openai_call(messages: List[Dict[str, str]], 
                            model: str, 
                            temperature: float, 
                            max_tokens: int, 
                            n: int, 
                            stop: Optional[List[str]] = None) -> Dict:
    """Make async call to OpenAI API with retry logic"""
    logger.debug(f"Message {messages} #END")
    response =  openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop
    ) #### await + acreate
    logger.debug(f"# Response {messages} #END")
    return response

async def chatgpt(messages: List[Dict[str, str]], 
                 model: str = "gpt-4o", 
                 temperature: float = 1.0, 
                 max_tokens: int = 100, 
                 n: int = 1, 
                 stop: Optional[List[str]] = None) -> List[str]:
    """Async function to interact with ChatGPT models"""
    global completion_tokens, prompt_tokens
    outputs = []
    
    try:
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            
            # Special handling for gpt-4o and gpt-4o mini
            if model in ["gpt-4o", "gpt-4o-mini"]:
                logger.debug(f"Using {model} configuration")
                # Adjust parameters based on the specific model
                if model == "gpt-4o":
                    temperature = min(temperature, 0.7)  # Cap temperature for gpt-4o
                elif model == "gpt-4o-mini":
                    temperature = min(temperature, 0.5)  # Cap temperature for gpt-4o mini
                # You can add more parameter adjustments if needed
                
            response = await asyncio.wait_for(
                async_openai_call(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=cnt,
                    stop=stop
                ),
                timeout=30
            )
            
            outputs.extend([choice["message"]["content"] for choice in response["choices"]])
            completion_tokens += response["usage"]["completion_tokens"]
            prompt_tokens += response["usage"]["prompt_tokens"]
            
        return outputs
        
    except asyncio.TimeoutError:
        logger.error("OpenAI API call timed out")
        raise TimeoutError("OpenAI API call timed out after 30 seconds")
    except Exception as e:
        logger.error(f"Error in chatgpt call: {str(e)}")
        raise

async def gpt4o(prompt: str, 
              model: str = "gpt-4o", 
              temperature: float = 0.2, 
              max_tokens: int = 100, 
              n: int = 1, 
              stop: Optional[List[str]] = None) -> List[str]:
    """Specialized async function for GPT-4 interactions with gpt-4o and gpt-4o mini support"""
    logger.debug(f"{model} prompt: {prompt} #END")
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        # Special handling for gpt-4o and gpt-4o mini requests
        if model in ["gpt-4o", "gpt-4o mini"]:
            logger.debug(f"Using {model} configuration")
            # Adjust configuration for gpt-4o and gpt-4o mini
            temperature = min(temperature, 0.7) if model == "gpt-4o" else min(temperature, 0.5)
            max_tokens = min(max_tokens, MAX_TOKENS)  # Ensure within limits
        
        response = await chatgpt(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )
        
        logger.debug(f"##gpt4o response: {response} #END")
        return response
        
    except Exception as e:
        logger.error(f"Error in gpt4o call: {str(e)}")
        raise

async def llm(prompt: str, 
             model: str = "gpt-4o", 
             temperature: float = 1.0, 
             max_tokens: int = 100, 
             n: int = 1, 
             stop: Optional[List[str]] = None) -> List[str]:
    """Main async interface for all LLM models"""
    try:
        if model.startswith("gpt"):
            return await gpt4o(prompt, model, temperature, max_tokens, n, stop)
        elif model.startswith("claude"):
            return await claude(prompt, model, temperature, max_tokens, n, stop)
        elif model.startswith("gemini"):
            return await gemini(prompt, model, temperature, max_tokens, n, stop)
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        logger.error(f"Error in llm call: {str(e)}")
        raise

def gpt_usage(backend: str = "gpt-4o") -> Dict[str, Any]:
    """Calculate token usage and cost"""
    global completion_tokens, prompt_tokens
    
    # Map gpt-4o and gpt-4o mini to appropriate cost structure
    if backend in ["gpt-4o", "gpt-4o-mini"]:
        backend = "gpt-4o"  # Assuming same cost structure for both variants
    
    cost_mapping = {
        "gpt-4o": (0.06, 0.03),  # (completion_cost, prompt_cost) per 1k tokens
        "gpt-4o-mini": (0.04, 0.02),  # Example costs for gpt-4o mini
        "gpt-3.5-turbo": (0.002, 0.0015),
        "gpt-3.5-turbo-16k": (0.004, 0.003)
    }
    
    if backend in cost_mapping:
        completion_cost, prompt_cost = cost_mapping[backend]
        cost = (completion_tokens / 1000 * completion_cost + 
                prompt_tokens / 1000 * prompt_cost)
    else:
        cost = 0
    
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost
    }