import sys
sys.path.append("../")
import os

# from openai import OpenAI
from google import genai
import tiktoken
from path_config import API_KEY_FILEPATH, PRO_DIR
from utils.context_util import Prompts
from utils.file_util import read_file

from traceback import print_exc

import time
GEMINI_KEY_FILEPATH = os.path.join(PRO_DIR, "gemini_api.key")

client = genai.Client(
    api_key=read_file(GEMINI_KEY_FILEPATH).strip()
)
# client = OpenAI(api_key=read_file(API_KEY_FILEPATH).strip())
# Organization IDs
# orgs = ["org-0wLi1kKIt9USgXMYklRCeTFQ", "org-9WS8eYC3IjH69yYgQNrk0X4w", "org-yclg2hcASx6eAd3nEyoFKrlf"]


# Setup                                                                                                                                                                                                    
 # OpenAPI key

MODEL_NAME = "gemini-flash-latest"

MODEL_MAX_TOKEN = {
    "gemini-flash-latest": 1048576
}

def oracle_prompt_generator(interact_index=None, context=None, summaries=None, example_method=None, setup=None, test=None, focal=None):
    if interact_index is None: raise Exception("In oracle prompt generator: interact_index cannot be None")
    if context is None: raise Exception("In oracle prompt generator: context cannot be None")
    if setup is None: setup = ""
    if test is None: raise Exception("In oracle prompt generator: test prefix cannot be None")
    if focal is None: raise Exception("In oracle prompt generator: focal method cannot be None")

    if interact_index == 0:
        # Insert seed message and a mock response( based on tests on ChatGPT WebUI ) in the oracle generation conversation
        context.insert(role="user", content=Prompts.GENERATION_SEED)                                        # INDX 1 (Ref. INDX constants in Context class in context_util.py)
        
        context.insert(role="assistant", content=Prompts.GENERATION_MOCK_SEED_RESPONSE)                     # INDX 2 (Ref. INDX constants in Context class in context_util.py)
        
        if summaries is not None:
            context.insert(role="user", content=(Prompts.GENERATION_SUMM_SEED).format(summaries))           # INDX 3 (Ref. INDX constants in Context class in context_util.py)
        else:
            context.insert(role="user", content="")                                                         # INDX 3 (Inserting a dummy empty message to maintain context history indices (which are important for resetting))
        
        if example_method is not None:
            context.insert(role="user", content=(Prompts.GENERATION_ONESHOT_SEED).format(example_method))   # INDX 4 (Ref. INDX constants in Context class in context_util.py)
        else:
            context.insert(role="user", content="")                                                         # INDX 4 (Inserting a dummy empty message to maintain context history indices (which are important for resetting))
        
        # Insert (after formatting) the prompt containing the setup method <SETUP>, test method prefix <TEST>, and focal method <FOCAL>
        context.insert(role="user", content=(Prompts.GENERATION_SEED_EXT).format(setup, test, focal))       # INDX 5 (Ref. INDX constants in Context class in context_util.py)
    else:
        context.insert(role="user", content=Prompts.GENERATION_NEXT)



def interact_with_openai_manual(temperature=1, context=None):
    import os
    import time

    if context is None:
        raise Exception("Context cannot be None")

    manual_dir = os.path.join(PRO_DIR, "manual_llm")
    os.makedirs(manual_dir, exist_ok=True)

    prompt_file = os.path.join(manual_dir, "prompt.txt")
    response_file = os.path.join(manual_dir, "response.txt")

    # Save the prompt
    with open(prompt_file, "w", encoding="utf-8") as f:
        for msg in context.history:
            f.write(f"{msg['role'].upper()}:\n")
            f.write(msg["content"])
            f.write("\n\n")

    print("\n====================================")
    print("Prompt written to:")
    print(prompt_file)
    print("Paste it into your chosen LLM.")
    print("Save the model response into:")
    print(response_file)
    print("Waiting for response...")
    print("====================================\n")

    # Remove old response if it exists
    if os.path.exists(response_file):
        os.remove(response_file)

    while not os.path.exists(response_file):
        time.sleep(2)

    while os.path.getsize(response_file) == 0:
        time.sleep(1)

    with open(response_file, "r", encoding="utf-8") as f:
        result = f.read().strip()

    os.remove(response_file)

    return result

def interact_with_gemini(temperature=1, context=None):

    if context is None:
        raise Exception("Context cannot be None")

    prompt = ""

    for msg in context.history:
        prompt += f"{msg['role'].upper()}:\n{msg['content']}\n\n"

    retries = 8

    for attempt in range(retries):
        try:
            print(f"Prompt length: {len(prompt)} characters")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            return response.text

        except Exception as e:

            print(f"\n===== GEMINI ERROR ({attempt + 1}/{retries}) =====")
            print(e)

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                raise Exception(
                    "Gemini free-tier quota exhausted. "
                    "Switch USE_PROVIDER='manual' or wait for quota reset."
                )

            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print("Retrying in 20 seconds...\n")
                time.sleep(20)
                continue

            raise

    raise Exception("Gemini failed after all retries.")

USE_PROVIDER = "gemini"    # "manual" or "gemini"

def interact_with_openai(temperature=1, context=None):
    if USE_PROVIDER == "manual":
        return interact_with_openai_manual(
            temperature=temperature,
            context=context
        )

    elif USE_PROVIDER == "gemini":
        return interact_with_gemini(
            temperature=temperature,
            context=context
        )

    raise ValueError(f"Unknown provider: {USE_PROVIDER}")

def num_tokens_from_string(query, model_name=MODEL_NAME):
    if USE_PROVIDER == "gemini":
        return len(query) // 4   # rough approximation

    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(query))

def if_exceed_token_limit(query, model_name=MODEL_NAME):
    """Returns True if the number of tokens in a text string exceeds the maximum allowed."""
    if model_name not in MODEL_MAX_TOKEN:
        raise ValueError(f"Model name {model_name} not supported.")
    max_tokens = MODEL_MAX_TOKEN[model_name]
    num_tokens = num_tokens_from_string(query, model_name)
    return num_tokens > max_tokens

def cut_query(query, model_name):
    """Cuts a query to the maximum number of tokens allowed by the model."""
    encoding = tiktoken.encoding_for_model(model_name)
    max_tokens = MODEL_MAX_TOKEN[model_name]
    num_tokens = len(encoding.encode(query))
    if num_tokens > max_tokens:
        query = encoding.decode(encoding.encode(query)[:max_tokens])
    return query
