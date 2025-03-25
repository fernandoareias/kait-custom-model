"""Entry point for CLI."""
import os
from datetime import datetime
import click
from kait.kubernetes_debugger import KubernetesDebugger
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential


@click.group()
def cli():
    """CLI wrapper."""


@cli.command()
@click.option("--api-key", required=True, help="API key for Azure OpenAI.")
@click.option("--endpoint", required=True, help="Azure OpenAI endpoint.")
@click.argument("prompt", default="Say hello")
def direct_test(api_key, endpoint, prompt):
    """Test directly with Azure OpenAI API bypassing the agent structure."""
    print(f"Testing direct integration with Azure OpenAI API")
    print(f"Endpoint: {endpoint}")
    print(f"Prompt: {prompt}")
    
    # Extract model name from endpoint URL
    import re
    match = re.search(r"/deployments/([^/]+)/?$", endpoint)
    if not match:
        print("Error: Invalid endpoint URL. Expected format: .../deployments/model-name")
        return
    
    model_name = match.group(1)
    print(f"Using model: {model_name}")
    
    try:
        # Create client
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        
        # Build user message
        messages = [
            UserMessage(content=f"""
You are an expert Kubernetes administrator and your job is to resolve the issue
described below using kubectl. You are already authenticated with the cluster.

{prompt}

Diagnose and fix the issue using kubectl.
""")
        ]
        
        # Send request
        print("Sending request to Azure OpenAI API...")
        response = client.complete(
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            model=model_name,
        )
        
        # Print response
        print("\n\n=== RESPONSE FROM AZURE OPENAI ===")
        print(response.choices[0].message.content)
        print("================================\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())


@cli.command()
@click.option("--api-key", required=True, help="API key for Azure OpenAI.")
@click.option("--endpoint", required=True, help="Azure OpenAI endpoint.")
def test_api(api_key, endpoint):
    """Test connectivity with Azure OpenAI API."""
    print(f"Testing connection to {endpoint}")
    
    # Extract model name from endpoint URL
    import re
    match = re.search(r"/deployments/([^/]+)/?$", endpoint)
    if not match:
        print("Error: Invalid endpoint URL. Expected format: .../deployments/model-name")
        return
    
    model_name = match.group(1)
    print(f"Using model: {model_name}")
    
    try:
        # Create client
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        
        # Test simple completion
        print("Sending test request...")
        response = client.complete(
            messages=[UserMessage(content="Say hello")],
            max_tokens=10,
            temperature=0.7,
            top_p=0.95,
            model=model_name,
        )
        
        print("Response received successfully!")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"Error connecting to Azure OpenAI: {str(e)}")
        import traceback
        print(traceback.format_exc())


@cli.command()
@click.argument("request")
@click.option(
    "--read-only",
    default=True,
    help="Flag for whether the command runs in read only mode or not. Defaults to true.",
)
@click.option(
    "--input-mode",
    type=click.Choice(["ALWAYS", "NEVER", "TERMINATE"], case_sensitive=False),
    default="NEVER",
    help="""
Defines whether kait should wait for human input or not (defaults to NEVER).
NEVER: Never prompt for user input;
ALWAYS: User will be requested for feedback after each step;
TERMINATE: User will be requested for feedback at the end of the debugging session.
""",
)
@click.option("--max-replies", type=int, default=10, help="Maximum number of replies before a session terminates.")
@click.option("--output-dir", help="Output directory to store the debug output markdown file.")
@click.option(
    "--policy",
    type=click.Choice(["autogen", "openai"], case_sensitive=False),
    default="autogen",
    help="""
The LLM policy to use (defaults to autogen).
autogen: Use AutoGen as the LLM provider;
openai: Use OpenAI directly as the LLM provider.
""",
)
@click.option("--api-key", help="API key for the LLM provider (required for OpenAI policy).")
@click.option("--endpoint", help="Azure OpenAI endpoint (required for OpenAI policy).")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging for debugging.")
@click.option("--timeout", type=int, default=60, help="Timeout in seconds for API calls. Default is 60.")
def debug(
    request,
    read_only,
    input_mode,
    max_replies,
    output_dir,
    policy,
    api_key,
    endpoint,
    verbose,
    timeout,
):
    """Debugs and fixes issues with kubernetes resources based on the provided REQUEST."""
    if policy == "openai" and (not api_key or not endpoint):
        print(
            """
The --api-key and --endpoint options are required when using the OpenAI policy.

Example:
    kait debug "your request" --policy openai --api-key YOUR_API_KEY --endpoint YOUR_ENDPOINT
"""
        )
        return

    if policy == "autogen" and os.getenv("KAIT_OPENAI_KEY") is None:
        print(
            """
The environment variable 'KAIT_OPENAI_KEY' is not set.

Set the environment variable 'KAIT_OPENAI_KEY' with your OpenAI config e.g.:
[
    {
        "model": "gpt-4",
        "api_key": "YOUR_OPENAI_API_KEY"
    }
]
"""
        )
        return

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_file: str = None
    if output_dir:
        file_date_key = datetime.today().strftime("%Y%m%d%H%M%S")
        output_file = f"{output_dir}/kait_debug_{file_date_key}.md"

        with open(output_file, encoding="UTF-8", mode="a") as file:
            file.write("## Request\n")
            file.write(f"{request}\n\n")
            file.write("## Response\n")

    kwargs = {}
    if policy == "openai":
        # Print debug info if verbose
        if verbose:
            print(f"Using OpenAI policy with endpoint: {endpoint}")
            print(f"API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}") 
        
        kwargs.update(
            {
                "api_key": api_key,
                "endpoint": endpoint,
                "verbose": verbose,
                "timeout": timeout,
            }
        )

    try:
        debugger = KubernetesDebugger(
            read_only=read_only,
            output_file=output_file,
            input_mode=input_mode,
            max_replies=max_replies,
            policy=policy,
            **kwargs,
        )
        debugger.debug(request=request)
    except Exception as e:
        print(f"Error in debug command: {str(e)}")
        if verbose:
            import traceback
            print(traceback.format_exc())


@cli.command()
@click.argument("request")
@click.option("--api-key", required=True, help="API key for Azure OpenAI.")
@click.option("--endpoint", required=True, help="Azure OpenAI endpoint.")
def simple_debug(request, api_key, endpoint):
    """Simple debug command that directly calls Azure OpenAI API."""
    direct_test(api_key, endpoint, request)


if __name__ == "__main__":
    cli()
