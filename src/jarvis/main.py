import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are Jarvis, a personal AI assistant running locally on the user's Windows computer.

Rules:
- Be concise and helpful.
- Do not claim you performed a computer action unless a tool actually performed it.
- Never invent file paths, messages, contacts, or system results.
- When an action is unavailable, explain that clearly.
"""

conversation = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def ask_jarvis(message: str) -> str:
    conversation.append({"role": "user", "content": message})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": conversation,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    answer = data["message"]["content"]
    conversation.append({"role": "assistant", "content": answer})

    return answer


def main():
    print("=" * 50)
    print("JARVIS LOCAL AI")
    print(f"Model: {MODEL}")
    print("Type 'exit' to stop Jarvis.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "/bye"}:
            print("\nJarvis: Goodbye.")
            break

        try:
            answer = ask_jarvis(user_input)
            print(f"\nJarvis: {answer}")
        except requests.exceptions.ConnectionError:
            print("\nJarvis: Ollama is not reachable.")
        except Exception as exc:
            print(f"\nJarvis error: {exc}")


if __name__ == "__main__":
    main()