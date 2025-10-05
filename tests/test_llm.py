import time
from agentine.llm import LLM, Message

def test_llm_init(provider: str = "openai") -> float:
    start = time.perf_counter()
    llm = LLM(provider=provider)
    end = time.perf_counter()
    elapsed = end - start
    print(f"LLM init for provider '{provider}' took {elapsed:.4f} seconds")
    return elapsed

def test_llm_switch(initial_provider: str = "openai", new_provider: str = "groq") -> float:
    llm = LLM(provider=initial_provider)
    print(f"Initialized with provider '{initial_provider}' and model '{llm.model}'")

    start = time.perf_counter()
    llm.provider = new_provider  # triggers __setattr__ and re-init of clients
    end = time.perf_counter()

    elapsed = end - start
    print(f"Switched to provider '{new_provider}' with model '{llm.model} {llm.base_url} {llm.api_key}' in {elapsed:.4f} seconds")
    return elapsed

# Test OpenAI
test_llm_init("openai")

# Test Groq
test_llm_init("groq")

# Test Anthropic
test_llm_init("anthropic")

# Switch from OpenAI to Groq
test_llm_switch("openai", "google")

# Switch from Groq to Anthropic
test_llm_switch("groq", "openai")

# Switch from OpenAI to DeepSeek
test_llm_switch("openai", "deepseek")

# llm = LLM()
# llm.provider = "google"
# # print("New config:")
# # print(llm)
# print(llm.chat([Message(content="Who is the first person walking on the moon?")]))