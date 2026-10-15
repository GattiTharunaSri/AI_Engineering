"""Ask an OpenAI model a question from the terminal and print its answer."""
from dotenv import load_dotenv
import openai

load_dotenv()  # loads OPENAI_API_KEY from a .env file next to this script

client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment

MODEL = "gpt-5"


def ask(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    question = input("Ask a question: ")
    answer = ask(question)
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
