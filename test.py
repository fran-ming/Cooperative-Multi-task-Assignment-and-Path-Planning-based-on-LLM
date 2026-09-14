import sys

from llm.client import LLMClient


if __name__ == "__main__":
    print("=== LLM connection test ===")
    print("Attempting to initialize client...")
    client = LLMClient()

    print(f"Model: {client.model_name}")
    print("Auth mode: API Key + Model Name")

    system_prompt = "You are a helpful assistant."
    user_prompt = "请直接回复：连接正常。"

    try:
        print("\nSending a simple request...")
        response = client.generate(system_prompt, user_prompt)
        print("\nResponse received successfully:")
        print(response[:400])
        print("\nStatus: OK")
        sys.exit(0)
    except Exception as e:
        print("\nStatus: FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        sys.exit(1)
