import ollama
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()

stream = ollama.chat(
    model="gemma4:e4b",
    messages=[
        {
            "role": "user",
            "content": "teach me the math and equation behind gradient decent",
        }
    ],
    stream=True,
)

buffer = ""
with Live(console=console, refresh_per_second=10) as live:
    for chunk in stream:
        buffer += chunk["message"]["content"]
        live.update(Markdown(buffer))
