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
@click.argument("message")
@click.option("--api-key", required=True, help="API key for Azure OpenAI.")
@click.option("--endpoint", required=True, help="Azure OpenAI endpoint.")
@click.option("--output-dir", help="Output directory to store the response file.")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging.")
@click.option("--timeout", type=int, default=60, help="Timeout in seconds for API calls. Default is 60.")
def direct_test(message, api_key, endpoint, output_dir, verbose, timeout):
    """
    Test direct API communication with Azure OpenAI.
    
    Sends the given MESSAGE directly to the Azure OpenAI API and returns the response.
    Useful for testing connectivity and response quality without using the agent framework.
    
    Examples:
        kait direct-test "How do I fix a CrashLoopBackOff in Kubernetes?" --api-key YOUR_KEY --endpoint YOUR_ENDPOINT
    """
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import UserMessage
    from azure.core.credentials import AzureKeyCredential
    import re
    
    print(f"Sending message to {endpoint}")
    
    # Prepare output file if directory specified
    output_file = None
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        file_date_key = datetime.today().strftime("%Y%m%d%H%M%S")
        output_file = f"{output_dir}/kait_direct_test_{file_date_key}.md"
        
        with open(output_file, encoding="UTF-8", mode="a") as file:
            file.write("# KAIT Direct API Test\n\n")
            file.write("## Query\n\n")
            file.write(f"{message}\n\n")
            file.write("## Response\n\n")
    
    try:
        # Extract model name from endpoint URL
        match = re.search(r"/deployments/([^/]+)/?", endpoint)
        if not match:
            print("❌ Invalid endpoint URL format")
            return
            
        model_name = match.group(1)
        
        if verbose:
            print(f"Using model: {model_name}")
            print(f"Timeout: {timeout} seconds")
        
        # Create client
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        
        if verbose:
            print("Sending request to Azure OpenAI API...")
            
        # Make the API call
        response = client.complete(
            messages=[UserMessage(content=message)],
            max_tokens=4000,
            temperature=0.7,
            top_p=0.95,
            model=model_name,
            timeout=timeout,
        )
        
        content = response.choices[0].message.content
        
        # Display response
        print("\n=== RESPONSE ===")
        print(content)
        print("===============\n")
        
        # Save to output file if provided
        if output_file:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{content}\n")
            print(f"📝 Response saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error communicating with Azure OpenAI API: {str(e)}")
        if verbose:
            import traceback
            print(traceback.format_exc())


@cli.command()
@click.argument("request")
@click.option(
    "--read-only",
    default=False,
    help="Flag for whether the command runs in read only mode or not. Defaults to false (allows write operations).",
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
@click.option("--direct", is_flag=True, default=True, help="Use direct API call instead of agent framework (default: True).")
@click.option("--execute", is_flag=True, default=False, help="Execute the suggested kubectl commands automatically (default: False).")
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
    execute,
):
    """
    Debugs and fixes issues with Kubernetes resources based on the provided REQUEST.

    KAIT (Kubernetes AI Tool) will analyze the problem, diagnose the issue, and 
    execute commands to fix it automatically.

    Examples:
        # Debug a service issue with the OpenAI policy
        kait debug "A Service in the hosting Namespace is not responding to requests" \\
          --policy openai --api-key YOUR_API_KEY --endpoint YOUR_AZURE_OPENAI_ENDPOINT

        # Debug a pod that keeps crashing
        kait debug "Pod my-app-pod keeps crashing with CrashLoopBackOff" \\
          --policy openai --api-key YOUR_API_KEY --endpoint YOUR_AZURE_OPENAI_ENDPOINT
    
    Note:
        The --read-only flag (default: false) restricts KAIT to only use commands that
        don't modify the cluster. Set to false to allow KAIT to make changes.
    """
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
            file.write("# KAIT Debugging Session\n\n")
            file.write("## Problem Statement\n\n")
            file.write(f"{request}\n\n")
            file.write("## Analysis and Resolution\n\n")

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
                "execute": execute,  # Pass execute flag to KubernetesDebugger
            }
        )

    try:
        print(f"⚡ KAIT is diagnosing: '{request[:100]}{'...' if len(request) > 100 else ''}'")
        if read_only:
            print("🔒 Running in read-only mode (no cluster modifications allowed)")
        else:
            print("✏️  Running in write mode (cluster modifications allowed)")
            
        debugger = KubernetesDebugger(
            read_only=read_only,
            output_file=output_file,
            input_mode=input_mode,
            max_replies=max_replies,
            policy=policy,
            **kwargs,
        )
        
        print("🔍 Starting debugging session...")
        result = debugger.debug(request=request)
        print("✅ KAIT has completed the debugging session")
        
        if output_file:
            print(f"📝 Complete analysis saved to: {output_file}")
        
        return result
    except Exception as e:
        print(f"❌ Error in debug command: {str(e)}")
        if verbose:
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    cli()
