import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
QUESTION = "What is 2 + 2? Answer with a single number."

TARGETS = [
    ("OpenRouter", "https://openrouter.ai/api/v1",
     os.getenv("OPENROUTER_API_KEY"), "openai/gpt-4o-mini"),
    ("Ollama", "http://localhost:11434/v1",
     "ollama", "llama3.2:3b"),
]

for name, base_url, key, model in TARGETS:
    if not key:
        print(f"{name}: OPENROUTER_API_KEY is missing in .env")
        continue
    client = OpenAI(base_url=base_url, api_key=key,
                    timeout=60, max_retries=2)
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": QUESTION}],
            temperature=0,
        )
    except Exception as e:
        print(f"{name}: error {type(e).__name__}: {e}")
        continue
    sec = time.perf_counter() - start
    u = r.usage
    answer = r.choices[0].message.content.strip()
    print(f"{name} | {model} | {sec:.1f}s | "
          f"tokens: in {u.prompt_tokens}, out {u.completion_tokens} | {answer}")
