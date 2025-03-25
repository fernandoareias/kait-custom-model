"""Entry point for CLI."""
import os
from datetime import datetime
import click
from kait.kubernetes_debugger import KubernetesDebugger


@click.group()
def cli():
    """CLI wrapper."""


@cli.command()
@click.option("--api-key", required=True, help="API key for Azure OpenAI.")
@click.option("--endpoint", required=True, help="Azure OpenAI endpoint.")
def test_api(api_key, endpoint):
    """Test connectivity with Azure OpenAI API."""
    from kait.policies.openai_policy import OpenAIStrategy
    
    print(f"Testing connection to {endpoint}")
    try:
        strategy = OpenAIStrategy(api_key=api_key, endpoint=endpoint)
        is_connected = strategy.test_connection()
        if is_connected:
            print("✅ Successfully connected to Azure OpenAI API")
        else:
            print("❌ Failed to connect to Azure OpenAI API")
    except Exception as e:
        print(f"❌ Error connecting to Azure OpenAI API: {str(e)}")
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
    default="openai",
    help="""
The LLM policy to use (defaults to openai).
autogen: Use AutoGen as the LLM provider;
openai: Use OpenAI directly as the LLM provider.
""",
)
@click.option("--api-key", help="API key for the LLM provider (required for OpenAI policy).")
@click.option("--endpoint", help="Azure OpenAI endpoint (required for OpenAI policy).")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging for debugging.")
@click.option("--timeout", type=int, default=60, help="Timeout in seconds for API calls. Default is 60.")
@click.option("--direct", is_flag=True, default=False, help="Use direct API call instead of agent framework.")
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
    direct,
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
                "direct": direct,  # Pass direct flag to KubernetesDebugger
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


if __name__ == "__main__":
    cli()
