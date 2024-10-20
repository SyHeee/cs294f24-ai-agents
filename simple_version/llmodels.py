import os
import time
import random
from dotenv import load_dotenv

import openai
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

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.info("Warning: OPENAI_API_KEY is not set in .env file")

# Anthropic setup
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    logger.info("Warning: ANTHROPIC_API_KEY is not set in .env file")

# Google AI setup
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.info("Warning: GOOGLE_API_KEY is not set in .env file")
else:
    genai.configure(api_key=google_api_key)

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.OpenAIError,),
):
    """Retry a function with exponential backoff."""
    
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                
                delay *= exponential_base * (1 + jitter * random.random())
                
                time.sleep(delay)
            
            except Exception as e:
                raise e
    
    return wrapper

@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt3(prompt, model="text-davinci-002", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    outputs = []
    for _ in range(n):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop
        )
        outputs.append(response.choices[0].text.strip())
    return outputs

def gpt(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    response = []
    if model == "text-davinci-002":
        response = gpt3(prompt, model, temperature, max_tokens, n, stop)        
    else:
        messages = [{"role": "user", "content": prompt}]
        response = chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    return response

def gpt4(prompt, model="gpt-4o", temperature=0.2, max_tokens=100, n=1, stop=None) -> list:
    logger.debug(f"{model} prompt: {prompt} #END")
    response = []
    if model == "text-davinci-002":
        response =  gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        response =  chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    logger.debug(f"gpt4 response: {response}")
    return response

def chatgpt(messages, model="gpt-4o", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

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

def llm(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    if model.startswith("gpt"):
        return gpt(prompt, model, temperature, max_tokens, n, stop)
    elif model.startswith("claude"):
        return claude(prompt, model, temperature, max_tokens, n, stop)
    elif model.startswith("gemini"):
        return gemini(prompt, model, temperature, max_tokens, n, stop)
    else:
        raise ValueError(f"Unsupported model: {model}")

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    else:
        cost = 0  # For non-GPT models, we don't calculate cost
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
