import os
import sys
from dotenv import load_dotenv
from smolagents import LiteLLMModel
from agents.research_agent import create_research_agent
from agents.data_agent import create_data_agent
from agents.manager_agent import create_manager_agent      # ← add this

load_dotenv()

def make_model(model_id: str) -> LiteLLMModel:
    return LiteLLMModel(
        model_id=model_id,
        api_base=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

model = make_model("mistral/mistral/mistral-small-latest")

research_agent = create_research_agent(model)
data_agent     = create_data_agent(model)
manager        = create_manager_agent(                     # ← use it here
    model=model,
    managed_agents=[research_agent, data_agent],
)

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "Research the latest LLM benchmarks and summarize the top 3 models."
    result = manager.run(task)
    print("\n=== RESULT ===")
    print(result)