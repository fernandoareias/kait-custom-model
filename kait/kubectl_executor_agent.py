"""Agent for executing kubectl commands."""
from typing import Optional
import traceback

from autogen.agentchat import UserProxyAgent
from kait.policies import LLMStrategy


READ_ONLY_COMMANDS = ["get", "describe", "explain", "logs", "top", "events", "api-versions", "cluster-info"]
BLOCKING_COMMANDS = ["edit", "--watch", "-w"]


class KubectlExecutorAgent(UserProxyAgent):
    """An agent for running kubectl commands.

    If running in read only mode (default), it will prevent create, read,
    update or delete commands from being executed.
    """

    def __init__(self, *args, read_only=True, llm_strategy: Optional[LLMStrategy] = None, **kwargs):
        """Initialize the KubectlExecutorAgent.

        Args:
            *args: Arguments to pass to the parent class.
            read_only (bool, optional): Whether to run in read-only mode. Defaults to True.
            llm_strategy (Optional[LLMStrategy], optional): The LLM strategy to use. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        print("[DEBUG-KUBECTL] Initializing KubectlExecutorAgent")
        print(f"[DEBUG-KUBECTL] Args: {args}")
        print(f"[DEBUG-KUBECTL] read_only: {read_only}")
        print(f"[DEBUG-KUBECTL] llm_strategy type: {type(llm_strategy).__name__}")
        print(f"[DEBUG-KUBECTL] kwargs: {kwargs}")
        
        try:
            super().__init__(*args, **kwargs)
            print("[DEBUG-KUBECTL] Parent class initialized successfully")
            self.read_only = read_only
            self.llm_strategy = llm_strategy
            print("[DEBUG-KUBECTL] KubectlExecutorAgent initialized successfully")
        except Exception as e:
            print(f"[ERROR-KUBECTL] Error initializing KubectlExecutorAgent: {e}")
            print(traceback.format_exc())
            raise

    def initiate_chat(self, *args, **kwargs):
        """Override initiate_chat to add logging"""
        print(f"[DEBUG-KUBECTL] Initiating chat with args: {args} and kwargs: {kwargs}")
        try:
            result = super().initiate_chat(*args, **kwargs)
            print("[DEBUG-KUBECTL] Chat initiated successfully")
            return result
        except Exception as e:
            print(f"[ERROR-KUBECTL] Error initiating chat: {e}")
            print(traceback.format_exc())
            raise

    def execute_code_blocks(self, code_blocks):
        """Execute kubectl command code blocks and returns the result.

        Args:
        ----
        code_blocks (list): kubectl commands to execute.

        Returns:
        -------
        A tuple of (exitcode, logs_all).
            exitcode (int): 0 if the code execution was successful, else non-zero.
            logs_all (str): The output of the code execution.
        """
        print(f"[DEBUG-KUBECTL] Executing code blocks: {code_blocks}")
        exitcode = 0
        logs_all = ""

        for code_block in code_blocks:
            lang, code = code_block
            print(f"[DEBUG-KUBECTL] Executing: {lang} code: {code}")

            if lang not in ("bash", "sh", "shell"):
                print(f"[DEBUG-KUBECTL] Unsupported language: {lang}")
                continue

            code = code.strip()

            if not code.startswith("kubectl"):
                error_msg = f"'{code}' is not a kubectl command."
                print(f"[ERROR-KUBECTL] {error_msg}")
                return 1, error_msg

            if any(command in code for command in (BLOCKING_COMMANDS)):
                error_msg = f"You cannot use the following commands/options, {BLOCKING_COMMANDS}, as they block execution."
                print(f"[ERROR-KUBECTL] {error_msg}")
                return (
                    1,
                    error_msg,
                )

            if self.read_only:
                if not any(command in code.split()[:2] for command in (READ_ONLY_COMMANDS)):
                    error_msg = f"\n'{code}' is not a read only operation. You can only perform read only operations.\n"
                    print(f"[ERROR-KUBECTL] {error_msg}")
                    return (
                        1,
                        error_msg,
                    )

            print(f"[DEBUG-KUBECTL] Running code: {code}")
            try:
                exitcode, logs, image = self.run_code(code, lang="bash", **self._code_execution_config)
                print(f"[DEBUG-KUBECTL] Code execution result: exitcode={exitcode}, logs={logs[:100]}...")
            except Exception as e:
                error_msg = f"Error executing kubectl command: {e}"
                print(f"[ERROR-KUBECTL] {error_msg}")
                print(traceback.format_exc())
                return 1, error_msg

            if image is not None:
                self._code_execution_config["use_docker"] = image
            logs_all += "\n" + logs
            if exitcode != 0:
                print(f"[ERROR-KUBECTL] Command failed with exitcode: {exitcode}")
                return exitcode, logs_all

        print(f"[DEBUG-KUBECTL] All code blocks executed successfully")
        return exitcode, logs_all
