import ollama

response = ollama.chat(
    model="gemma4:e4b",
    messages=[
        {
            "role": "system",
            "content": "Your are a query generator, you will generate multiple / suitable search queries that"
            + " can be used to find the answer to the user's question. give it in json format with a single key 'query' and the value being the search query."
            + 'Out put must be exactly as follows: {"query": "search query 1 here", "query": "search query 2 here"} (max up to 5 quries) and nothing else. Do not add any other text or explanation.',
        },
        {
            "role": "user",
            "content": "tell me everythin I need to know about music and machine learning related to music generation and music classification",
        },
    ],
    stream=True,
    keep_alive="1m",
    think="low",
)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
