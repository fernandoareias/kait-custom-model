![Python Version](https://img.shields.io/badge/3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.13-blue)

# Kait (Kubernetes AI Tool)

`kait` is an AI assisted kubernetes debugging tool which automatically troubleshoots, diagnoses, and fixes issues related to kubernetes.

`kait` uses autonomous AI agents built using Microsoft's [AutoGen](https://microsoft.github.io/autogen/)

## Installation and setup

You can install `kait` directly using pip:

```bash
pip install kait
```

`kait` requires an OpenAI API key which is read via the environment variable KAIT_OPENAI_KEY. You can provide a list of models to use and `kait` will use the available model. It is recommended to use models with the capabilities of gpt-4. Larger contexts work better too e.g. 'gpt-4-1106-preview'. Your environment variable needs to point to a list of models in the following format:

```
[
    {
        "model": "gpt-4",
        "api_key": "YOUR_OPENAI_API_KEY"
    }
]
```

### Using a Local LLM

You can use OpenAI compatible local LLMs by including a `base_url` in your model spec:

```
[
    {
        "model": "chatglm2-6b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "NULL", # Any string will do
    }
]
```

[FastChat/](https://github.com/lm-sys/FastChat) and [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/) both provide OpenAI compatible APIs which can be used with the above config. Which models provide adequate performance still needs validating.

## Local Development

If you want to contribute or run the project locally for development, follow these steps:

### Prerequisites

- Python 3.8 or higher
- Git
- A Kubernetes cluster for testing (optional)
- Azure OpenAI API key or other LLM API access

### Setup Local Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kait-custom-model.git
   cd kait-custom-model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file for local development:
   ```bash
   touch .env
   ```

5. Add your Azure OpenAI API key and endpoint to the `.env` file:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name
   ```

### Running the Tool Locally

After setting up your environment, you can run the tool with various commands:

1. Test API connectivity:
   ```bash
   python -m kait test-api --api-key $AZURE_OPENAI_API_KEY --endpoint $AZURE_OPENAI_ENDPOINT
   ```

2. Run a direct test (no agents):
   ```bash
   python -m kait direct-test "How do I fix a CrashLoopBackOff in Kubernetes?" --api-key $AZURE_OPENAI_API_KEY --endpoint $AZURE_OPENAI_ENDPOINT
   ```

3. Debug a Kubernetes issue:
   ```bash
   python -m kait debug "A pod in the default namespace is in CrashLoopBackOff state" --policy openai --api-key $AZURE_OPENAI_API_KEY --endpoint $AZURE_OPENAI_ENDPOINT --verbose
   ```

4. Debug and execute commands automatically:
   ```bash
   python -m kait debug "A service is not responding" --policy openai --api-key $AZURE_OPENAI_API_KEY --endpoint $AZURE_OPENAI_ENDPOINT --execute
   ```

### Running Tests

To run the test suite:

```bash
pytest
```

## Usage

`kait` requires kubectl to be installed and authenticated against the cluster you want to use.

To run `kait` simply run:

```bash
kait debug <DESCRIPTION OF THE ISSUE TO DEBUG>
```

For a full list of options, run:

```bash
kait debug --help
```

## Examples

A number of [examples/](examples/README.md) are provided so you can see how `kait` performs.
