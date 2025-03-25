"""Agent framework for debugging kubernetes."""
from typing import Optional
import traceback

import autogen
from colorama import Fore, Style

from kait.kubectl_executor_agent import KubectlExecutorAgent
from kait.policies import get_llm_strategy


class KubernetesDebugger:
    """Kubernetes debugger agent framework.

    The kubernetes debugging framework consisting of:
    - A kubernetes admin agent who performs the debugging
    - A code executor agent who runs the code suggested by the kubernetes admin

    These agents work together to solve the problem outlined by the user.
    """

    def __init__(
        self,
        read_only: bool,
        output_file: Optional[str] = None,
        input_mode: str = "NEVER",
        max_replies: int = 10,
        policy: Optional[str] = None,
        verbose: bool = False,
        direct: bool = False,
        execute: bool = False,
        **kwargs,
    ):
        """Initialize the KubernetesDebugger.

        Args:
            read_only (bool): Whether to run in read-only mode.
            output_file (Optional[str], optional): The output file to write to. Defaults to None.
            input_mode (str, optional): The input mode. Defaults to "NEVER".
            max_replies (int, optional): The maximum number of replies. Defaults to 10.
            policy (Optional[str], optional): The LLM policy to use. Defaults to None.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            direct (bool, optional): Whether to use direct API call instead of agent framework. Defaults to False.
            execute (bool, optional): Whether to execute the suggested commands. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the LLM strategy.
        """
        print("[DEBUG-KUBE] Initializing KubernetesDebugger")
        self.read_only = read_only
        self.output_file = output_file
        self.input_mode = input_mode
        self.max_replies = max_replies
        self.verbose = verbose or kwargs.get('verbose', False)
        self.direct = direct or kwargs.get('direct', False)
        self.execute = execute or kwargs.get('execute', False)

        if self.verbose:
            print(f"[DEBUG-KUBE] Getting LLM strategy for policy: {policy}")
        self.llm_strategy = get_llm_strategy(policy, **kwargs)
        if self.verbose:
            print("[DEBUG-KUBE] LLM strategy obtained")
        
        self.config_list = self.llm_strategy.get_config_list()
        if self.verbose:
            print(f"[DEBUG-KUBE] Config list: {self.config_list}")
            
        # If using direct mode, we don't need to set up the agents
        if not self.direct:
            if self.verbose:
                print("[DEBUG-KUBE] Setting up kubernetes admin agent")
            self.kubernetes_agent = self._setup_kubernetes_admin_agent()
            if self.verbose:
                print("[DEBUG-KUBE] Kubernetes admin agent setup completed")
            
            if self.verbose:
                print("[DEBUG-KUBE] Setting up code executor agent")
            self.code_agent = self._setup_code_executor_agent()
            if self.verbose:
                print("[DEBUG-KUBE] Code executor agent setup completed")
                
        if self.verbose:
            print("[DEBUG-KUBE] KubernetesDebugger initialized successfully")

    def _setup_kubernetes_admin_agent(self):
        """Set up the kubernetes admin agent.

        Returns:
            Agent: The kubernetes admin agent.
        """
        print("[DEBUG-KUBE] Inside _setup_kubernetes_admin_agent")
        if self.read_only:
            read_only_clause = """
You only have read only permissions to the cluster. You cannot create, update or delete
resources on the cluster, but you can use the following commands; 'get', 'describe', 'logs', 'top', 'events'.
You can propose changes for the end user to run to fix the command afterwards.
"""
        else:
            read_only_clause = ""

        print("[DEBUG-KUBE] Creating kubernetes_agent with llm_strategy")
        try:
            kubernetes_agent = self.llm_strategy.create_agent(
            name="kait",
            system_message=f"""
# Your Role
You are an expert Kubernetes administrator and your job is to resolve issues relating to
Kubernetes deployments using kubectl. Do not try and debug the issue with the containers themselves
and only focus on issues relating to kubernetes itself. You are already logged in to the cluster.


# How to present code
You must only provide one command to execute at a time. Never put placeholders like <pod-name> or
<service-name> in your code. Make sure you limit the output when running 'kubectl logs' using the
--tail or --since flags; if using --tail limit logs to the last 10 records.
Do not use the 'kubectl edit' command.


{read_only_clause}


# Useful kubectl command
get: Get resources deployed to the cluster.
describe: Provides details about resources deployed on the cluster.
logs: Get Pod logs.
top: Show CPU and memory metrics.
events: Lists all warning events.
apply: Declaratively create resources on the cluster.
create: Imperatively create resources on the cluster.
patch: Partially update a resource
expose: Create a service


# How to terminate the debug session
Don't ask if the user needs any further assistance, simply reply with 'TERMINATE' if you
have completed the task to the best of your abilities. If you need the user to save code to a file
or you have no code left to run, respond with 'TERMINATE'.
""",
            llm_config={
                "timeout": 600,
                "cache_seed": 42,
                "config_list": self.config_list,
                "temperature": 0,
            },
            code_execution_config={
                "use_docker": False,
            },
        )
            print("[DEBUG-KUBE] kubernetes_agent created successfully")
        except Exception as e:
            print(f"[ERROR-KUBE] Failed to create kubernetes_agent: {e}")
            print(traceback.format_exc())
            raise

        def output_message(recipient, messages, sender, config):
            print(f"[DEBUG-KUBE] Inside output_message: recipient={recipient.name}, sender={sender.name}")
            if "callback" in config and config["callback"] is not None:
                callback = config["callback"]
                callback(sender, recipient, messages[-1])

            if len(messages) == 1:
                return False, None

            content = messages[-1]["content"]
            if "Code output:" not in content:
                # No code was executed.
                return False, None

            console_output = f"{Fore.MAGENTA}{content.split('Code output:')[1]} {Style.RESET_ALL}"
            print(console_output)

            if self.output_file:
                file_output = f"\n\n```yaml{content.split('Code output:')[1]}```\n\n"
                with open(self.output_file, "a", encoding="utf-8") as file:
                    file.write(file_output)

            return False, None

        print("[DEBUG-KUBE] Registering reply for kubernetes_agent")
        kubernetes_agent.register_reply(
            [autogen.Agent, None],
            reply_func=output_message,
            config={"callback": None},
        )
        print("[DEBUG-KUBE] Reply registered for kubernetes_agent")

        return kubernetes_agent

    def _setup_code_executor_agent(self):
        """Set up the code executor agent.

        Returns:
            KubectlExecutorAgent: The code executor agent.
        """
        print("[DEBUG-KUBE] Inside _setup_code_executor_agent")
        try:
            code_executor_agent = KubectlExecutorAgent(
                read_only=self.read_only,
                name="user",
                human_input_mode=self.input_mode,
                max_consecutive_auto_reply=self.max_replies,
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                code_execution_config={
                    "use_docker": False,
                },
                llm_strategy=self.llm_strategy,
            )
            print("[DEBUG-KUBE] code_executor_agent created successfully")
        except Exception as e:
            print(f"[ERROR-KUBE] Failed to create code_executor_agent: {e}")
            print(traceback.format_exc())
            raise

        def output_message(recipient, messages, sender, config):
            print(f"[DEBUG-KUBE] Inside code_executor output_message: recipient={recipient.name}, sender={sender.name}")
            if "callback" in config and config["callback"] is not None:
                callback = config["callback"]
                callback(sender, recipient, messages[-1])

            content = messages[-1]["content"]
            if content.endswith("TERMINATE"):
                content = content.replace("TERMINATE", "")

            print(content)

            if self.output_file:
                with open(self.output_file, "a", encoding="utf-8") as file:
                    file.write(content)

            return False, None

        print("[DEBUG-KUBE] Registering reply for code_executor_agent")
        code_executor_agent.register_reply(
            [autogen.Agent, None],
            reply_func=output_message,
            config={"callback": None},
        )
        print("[DEBUG-KUBE] Reply registered for code_executor_agent")

        return code_executor_agent

    def debug(self, request):
        """
        Debugs and fixes kubernetes issues.

        Args:
            request (str): A string representing the problem to be fixed.

        Returns:
            None
        """
        if self.verbose:
            print("[DEBUG-KUBE] Inside debug method with request")
            
        # Format user request
        message = f"""
You are an expert Kubernetes administrator and your job is to resolve the issue
described below using kubectl. You are already authenticated with the cluster.

IMPORTANT: You must provide commands to fix the issue, one at a time. Each command should be in a code block.
Do not use placeholders like <pod-name> or <namespace>. Use real resource names.
For logs, limit output with --tail=10 to show only recent logs.

PROBLEM:
{request}

Diagnose and fix the issue by following these steps:
1. First, investigate the issue using read-only commands (get, describe, logs)
2. Identify the root cause of the problem
3. Apply the necessary fixes using kubectl commands
4. Verify your fix has resolved the issue

Provide your reasoning at each step and be thorough in your diagnosis.
"""
        if self.verbose:
            print(f"[DEBUG-KUBE] Initiating chat with user request: {request[:50]}...")
        
        # If direct mode is enabled, use direct API call
        if self.direct:
            return self._direct_debug(message)
        
        try:
            # Ensure the agent is properly set up
            if not hasattr(self, 'kubernetes_agent') or not hasattr(self, 'code_agent'):
                print("[ERROR-KUBE] Agents not properly initialized")
                raise RuntimeError("Agents not properly initialized")
                
            # Set system message for kubernetes agent if not already set
            if hasattr(self.kubernetes_agent, 'system_message') and not self.kubernetes_agent.system_message:
                self.kubernetes_agent.system_message = """
You are an expert Kubernetes administrator. Your job is to diagnose and fix Kubernetes issues.
Always be thorough in your analysis and provide clear explanations along with your commands.
"""
            
            # Initiate chat with clear instructions
            print("[DEBUG-KUBE] Starting agent conversation...")
            response = self.code_agent.initiate_chat(
                self.kubernetes_agent, 
                silent=not self.verbose,  # Make verbose mode actually show the conversation
                message=message,
            )
            
            if self.verbose:
                print(f"[DEBUG-KUBE] Chat completed with response: {str(response)[:100]}...")
            
            # Check if we got a valid response
            if not response or (isinstance(response, dict) and not response.get('chat_history')):
                print("[DEBUG-KUBE] No valid response from agent conversation, falling back to direct API call")
                direct_response = self._direct_debug(message)
                
                # If an output file is specified, write the direct response to it
                if self.output_file and direct_response:
                    try:
                        with open(self.output_file, "a", encoding="UTF-8") as file:
                            file.write("\n## Direct API Response\n\n")
                            file.write(direct_response)
                    except Exception as e:
                        print(f"[ERROR-KUBE] Failed to write direct response to output file: {str(e)}")
                
                return direct_response
            
            # If an output file is specified, write the conversation to it
            if self.output_file:
                try:
                    # Get chat history from response if available
                    chat_history = []
                    if hasattr(response, 'chat_history'):
                        chat_history = response.chat_history
                    elif hasattr(response, 'chat_id') and hasattr(response, 'chat_history'):
                        chat_history = response.chat_history
                    elif isinstance(response, dict) and 'chat_history' in response:
                        chat_history = response['chat_history']
                        
                    with open(self.output_file, "a", encoding="UTF-8") as file:
                        file.write("\n## Debugging Session\n\n")
                        
                        # If we have chat history, write it to the file
                        if chat_history:
                            for msg in chat_history:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "")
                                    content = msg.get("content", "")
                                    
                                    if role == "assistant" and content:
                                        file.write(f"### KAIT Analysis\n{content}\n\n")
                                    elif role == "user" and content and content.startswith("Code output:"):
                                        file.write(f"### Command Output\n```\n{content.replace('Code output:', '')}\n```\n\n")
                                        
                        # If no chat history, write a summary of the debug session
                        else:
                            file.write(f"### Debug Summary\n\nKAIT has analyzed the issue: '{request[:100]}...'\n\n")
                            
                    if self.verbose:
                        print(f"[DEBUG-KUBE] Results written to output file: {self.output_file}")
                except Exception as e:
                    print(f"[ERROR-KUBE] Failed to write to output file: {str(e)}")
            
            print("[DEBUG-KUBE] Debug session completed successfully")
            return response
            
        except Exception as e:
            print(f"[ERROR-KUBE] Error in debug: {str(e)}")
            if self.verbose:
                import traceback
                print(traceback.format_exc())
            
            # If there's an error, try a direct call to get a response
            print("[DEBUG-KUBE] Falling back to direct API call due to error")
            return self._direct_debug(message)
    
    def _direct_debug(self, message):
        """
        Make a direct API call to the LLM provider.
        
        Args:
            message (str): The message to send to the LLM provider.
            
        Returns:
            str: The response from the LLM provider.
        """
        if self.verbose:
            print("[DEBUG-KUBE] Using direct API call")
            
        # Check if we have a valid OpenAI strategy
        if self.llm_strategy.__class__.__name__ != "OpenAIStrategy" or not hasattr(self.llm_strategy, 'endpoint') or not hasattr(self.llm_strategy, 'api_key'):
            print("[ERROR-KUBE] Direct mode requires OpenAI strategy with valid endpoint and API key")
            return None
            
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.ai.inference.models import UserMessage, SystemMessage
            from azure.core.credentials import AzureKeyCredential
            import re
            
            endpoint = self.llm_strategy.endpoint
            api_key = self.llm_strategy.api_key
            
            # Extract model name from endpoint URL
            match = re.search(r"/deployments/([^/]+)/?", endpoint)
            if not match:
                print("[ERROR] Invalid endpoint URL format")
                return None
                
            model_name = match.group(1)
            
            if self.verbose:
                print(f"[DEBUG-KUBE] Making direct API call to {endpoint}")
                print(f"[DEBUG-KUBE] Using model: {model_name}")
            
            # Create client and make direct call
            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            
            if self.verbose:
                print("[DEBUG-KUBE] Sending request...")
            
            # Create a system message to give better instructions
            system_message = """You are an expert Kubernetes administrator. Your job is to diagnose and fix 
Kubernetes issues. 

Follow these guidelines:
1. Provide specific kubectl commands for diagnostics - no placeholders, use real command syntax
2. For each command, explain your reasoning in detail
3. Start with read-only commands (get, describe, logs) to diagnose
4. After diagnosis, provide specific commands to fix the issue
5. Include verification commands to confirm the fix worked
6. Format all commands in code blocks using markdown ```bash ... ``` format

Remember to be thorough, careful, and precise in your analysis and recommendations."""
                
            response = client.complete(
                messages=[
                    SystemMessage(content=system_message),
                    UserMessage(content=message)
                ],
                max_tokens=4000,
                temperature=0.7,
                top_p=0.95,
                model=model_name,
            )
            
            content = response.choices[0].message.content
            
            if self.verbose:
                print("[DEBUG-KUBE] Response received successfully")
            
            print("\n=== RESPONSE ===")
            print(content)
            print("===============\n")
            
            # Save to output file if provided
            if self.output_file:
                # Write to a temporary file first to capture LLM analysis
                with open(self.output_file, "r", encoding="utf-8") as f:
                    file_content = f.read()
                
                # Look for the Analysis and Resolution section to add the content after it
                if "## Analysis and Resolution" in file_content:
                    # Split the file at the Analysis and Resolution section
                    parts = file_content.split("## Analysis and Resolution")
                    
                    # Write back the first part + the analysis section + the model's response
                    with open(self.output_file, "w", encoding="utf-8") as f:
                        f.write(parts[0])  # Header + Problem Statement
                        f.write("## Analysis and Resolution\n\n")  # Analysis section title
                        f.write(content)  # LLM response
                else:
                    # Append to the end if section not found
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(content)
                
                if self.verbose:
                    print(f"[DEBUG-KUBE] Response saved to {self.output_file}")
            
            # Execute the suggested commands if not in read-only mode and execute is enabled
            if not self.read_only and self.execute:
                print("\n⚙️ Executing recommended commands...\n")
                self._execute_commands_from_response(content)
                    
            return content
        except Exception as e:
            print(f"[ERROR-KUBE] Failed to get direct response: {str(e)}")
            if self.verbose:
                import traceback
                print(traceback.format_exc())
            return None
    
    def _execute_commands_from_response(self, response_content):
        """
        Extract and execute kubectl commands from the model's response.
        
        Args:
            response_content (str): The content of the model's response.
            
        Returns:
            None
        """
        import re
        import subprocess
        import time
        from colorama import Fore, Style
        
        # Extract commands from code blocks marked with ```bash
        # Also support other formats like ```sh or just ```
        bash_code_blocks = re.findall(r'```(?:bash|sh|shell)?\s*(.*?)\s*```', response_content, re.DOTALL)
        
        if not bash_code_blocks:
            print("[DEBUG-KUBE] No kubectl commands found in the response")
            return
        
        # Initialize a list to store command execution results
        execution_results = []
        
        # Create a string to collect command outputs for the summary section
        command_outputs = "\n## OUTPUT Commands\n\n"
        
        # Execute each command in order
        for i, code_block in enumerate(bash_code_blocks):
            # Extract commands (might have multiple lines in a block)
            commands = [cmd.strip() for cmd in code_block.split('\n') if cmd.strip()]
            
            for cmd in commands:
                # Only execute kubectl commands for safety
                if not cmd.startswith('kubectl'):
                    print(f"[WARNING] Skipping non-kubectl command: {cmd}")
                    continue
                
                print(f"\n{Fore.CYAN}> Executing command: {cmd}{Style.RESET_ALL}")
                
                # Execute the command
                try:
                    start_time = time.time()
                    process = subprocess.Popen(
                        cmd, 
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = process.communicate()
                    execution_time = time.time() - start_time
                    
                    # Display the result
                    if process.returncode == 0:
                        print(f"{Fore.GREEN}✓ Command completed successfully in {execution_time:.2f}s{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}{stdout}{Style.RESET_ALL}")
                        # Add successful command to outputs
                        command_outputs += f"> {cmd}\n{stdout}\n\n"
                    else:
                        print(f"{Fore.RED}✗ Command failed with exit code {process.returncode} in {execution_time:.2f}s{Style.RESET_ALL}")
                        print(f"{Fore.RED}{stderr}{Style.RESET_ALL}")
                        # Add failed command to outputs
                        command_outputs += f"> {cmd}\n*Error:* {stderr}\n\n"
                    
                    # Store the result
                    execution_results.append((cmd, stdout if process.returncode == 0 else stderr, process.returncode))
                    
                    # Write to output file if provided - individual command execution
                    if self.output_file:
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            f.write(f"\n### Command Execution\n\n")
                            f.write(f"```bash\n{cmd}\n```\n\n")
                            f.write(f"**Result:**\n\n")
                            f.write(f"```\n{stdout if process.returncode == 0 else stderr}\n```\n\n")
                            
                except Exception as e:
                    print(f"{Fore.RED}Error executing command: {str(e)}{Style.RESET_ALL}")
                    execution_results.append((cmd, str(e), -1))
                    # Add error to outputs
                    command_outputs += f"> {cmd}\n*Error:* {str(e)}\n\n"
                    
                # Add a small delay between commands
                time.sleep(1)
        
        # Summary of command execution
        print(f"\n{Fore.CYAN}Command Execution Summary:{Style.RESET_ALL}")
        success_count = sum(1 for _, _, code in execution_results if code == 0)
        fail_count = sum(1 for _, _, code in execution_results if code != 0 and code != -1)
        skip_count = sum(1 for _, _, code in execution_results if code == -1)
        print(f"{Fore.GREEN}Successful: {success_count}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {fail_count}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Skipped: {skip_count}{Style.RESET_ALL}")
        
        # Write summary to output file if provided
        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                # Add the collected command outputs section after the analysis section
                f.write(command_outputs)
                
                # Add execution summary
                f.write(f"\n### Execution Summary\n\n")
                f.write(f"- Successful: {success_count}\n")
                f.write(f"- Failed: {fail_count}\n")
                f.write(f"- Skipped: {skip_count}\n\n")
