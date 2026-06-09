# physics_agent_harness

Experiment harness for agents, built on [smolagents](https://github.com/huggingface/smolagents).

## Prerequisites

- Python 3.10+
- Access to a LiteLLM-compatible endpoint (or a [Hugging Face account](https://huggingface.co/join))

## Installation

```bash
git clone https://github.com/your-username/physics_agent_harness.git
cd physics_agent_harness

python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# installs smolagents[toolkit], smolagents[litellm], and other dependencies
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
```

Open `.env` and fill in:

```bash
# LiteLLM endpoint
OPENAI_BASE_URL=http://your-litellm-host/v1
OPENAI_API_BASE=http://your-litellm-host/v1
OPENAI_API_KEY=your_api_key_here

# Or Hugging Face
HF_TOKEN=your_huggingface_token_here
```

> `.env` is in `.gitignore` and will never be committed.

## Usage

```bash
python run.py "Research the latest LLM benchmarks and summarize the top 3 models"
python run.py "Load data/results.csv and tell me which experiment had the highest accuracy"
```

## How agents work

Each agent runs a ReAct loop: think → act → observe → repeat until `final_answer()` is called.

There are two agent types. Both support the same tools — the difference is how the LLM expresses actions:

- `CodeAgent` — LLM writes Python code. Supports loops, conditions, combining results. Best for open-ended tasks. Uses `code_agent.yaml` as default prompt.
- `ToolCallingAgent` — LLM writes JSON tool calls. Safe and predictable. Best for fixed, simple tool calls. Uses `toolcalling_agent.yaml` as default prompt.

```python
# CodeAgent — LLM writes this
results = web_search("AI trends 2025")
final_answer(results)

# ToolCallingAgent — LLM writes this
{"tool": "web_search", "arguments": {"query": "AI trends 2025"}}
```

## Prompt YAML structure

Each agent has a prompt YAML with 4 sections. Each section is called at a different moment:

```yaml
system_prompt:       # always — the agent's main personality and instructions
planning:            # only if planning_interval is set on the agent
  initial_plan:      # runs at step 1 to produce a facts survey + plan
  update_plan_pre_messages:
  update_plan_post_messages:
managed_agent:       # only when this agent is called by a manager agent
  task:              # wraps the task string the manager sends down
  report:            # wraps the result sent back up to the manager
final_answer:        # only when the agent hits max_steps without finishing
  pre_messages:
  post_messages:
```

If you don't pass a `prompt_templates` when creating an agent, smolagents loads its own default (`code_agent.yaml` for `CodeAgent`, `toolcalling_agent.yaml` for `ToolCallingAgent`). The manager in this repo uses the default — only the worker agents have custom prompts.

The YAML key names are hardcoded in smolagents — do not rename them or you will get a `KeyError`.

## Configuring agents and models

All agent and model declarations are in `run.py`:

```python
research_agent = create_research_agent(model=make_model("mistral/mistral-small-latest"))
data_agent     = create_data_agent(model=make_model("mistral/mistral-small-latest"))

manager = CodeAgent(
    tools=[],
    model=make_model("mistral/mistral-small-latest"),
    managed_agents=[research_agent, data_agent],
)
```

`managed_agents` is a registration list — it tells the manager which agents exist. It does not control call order. The manager's LLM decides at runtime which agent to call and when based on the task.

### Controlling steps per agent

Each agent runs until it calls `final_answer()` or hits `max_steps`. The default is 20 steps for all agents. Lower it to reduce LLM calls, raise it for complex tasks:

```python
# set at agent creation
research_agent = create_research_agent(model=make_model("mistral/mistral-small-latest"))  # max_steps=5 set inside

# or override per run
manager.run("your task", max_steps=10)
```

Default values in smolagents:
- `max_steps=20` — every agent, set in `MultiStepAgent.__init__()`
- `planning_interval=None` — planning off by default
- `stream_outputs=False` — streaming off by default

If an agent hits `max_steps` without finishing, smolagents uses the `final_answer` section of the YAML as a fallback prompt to force a best-effort answer.

### Model ID format for LiteLLM

LiteLLM strips the provider prefix before forwarding to the server. Double the prefix so the server receives the correct ID:

```python
# Mistral — double the prefix
make_model("mistral/mistral/mistral-small-latest")   # works
make_model("mistral/mistral/mistral-large-latest")   # works
```

## Repo structure

```
physics_agent_harness/
├── agents/             one file per agent
├── prompts/            one yaml per agent — edit to tune behavior
├── tools/              custom tools
├── run.py              entry point
├── requirements.txt
└── .env.example
```

## Adding a new agent

1. `agents/my_agent.py` — define `create_my_agent(model)`
2. `prompts/my_agent.yaml` — write the system prompt
3. `run.py` — import and add to `managed_agents`

## Debugging

```python
result = manager.run("your task")
print(manager.logs)   # full step-by-step trace
```

If you see repeated `Import of X is not allowed` errors, add the import to `additional_authorized_imports` on the agent that needs it:

```python
# agents/data_agent.py or agents/manager_agent.py
CodeAgent(
    ...
    additional_authorized_imports=["pandas", "csv", "numpy"],
)
```
