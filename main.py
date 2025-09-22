from runtime import build_graph
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    app = build_graph()
    inputs = {"input": {"query": "Extract and analyze this: sample data", "params": {"max_tokens": 100}}}
    
    # Run preview
    for step in app.stream(inputs):
        for node, data in step.items():
            print(f"{node}: {data}")