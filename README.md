# viberesearcher
VibeResearcher is a PoC showing how agents based on LLMs can do automated model research.

## How it works
VibeResearcher uses a ReAct loop where an agent can save notes, write training data, and train models. When a model is finished training, the newly-trained model's results is given to the agent, and the agent adjusts the training data until you get a smart model.

## How to use
If you have dependencies installed, run main.py and create the files `notes.md` and `pairs.jsonl` if they don't exist yet.
If you don't have dependencies installed, install GPU accelerated versions of unsloth, datasets, accelerate, and ollama using PIP. You then need to install Ollama onto your computer and pull the model: `ollama pull fredrezones55/qwen3.5-opus:4b`

## Hardware
VibeResearcher has been tested and made for GPUs with low VRAM. VibeResearcher runs on GPUs with as low as 6 GB of VRAM.
