import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="製品マニュアルに関する質問:\n再起動するコマンドは？/n/n回答:",
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
)

answer = response.choices[0].text.strip()
print(answer)
