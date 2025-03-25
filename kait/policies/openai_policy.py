"""OpenAI LLM strategy implementation using Azure AI SDK."""
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import re
import traceback
import time
import sys
import json
import socket

from autogen.agentchat import Agent
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError, ClientAuthenticationError

from kait.policies.base import LLMStrategy


class OpenAIStrategy(LLMStrategy):
    """Strategy for using OpenAI directly as the LLM provider through Azure AI SDK."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        verbose: bool = False,
        timeout: int = 60,
    ):
        """Initialize the OpenAI strategy.

        Args:
            api_key (str): The Azure OpenAI API key.
            endpoint (str): The Azure OpenAI endpoint.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            timeout (int, optional): Timeout in seconds for API calls. Defaults to 60.
        """
        print("[INIT] Initializing OpenAIStrategy")
        self.api_key = api_key
        self.endpoint = endpoint
        self.verbose = verbose
        self.timeout = timeout
        
        if len(api_key) < 10:
            print("[WARNING] API key seems too short, might be invalid")
        print(f"[DEBUG] API Key (first/last 5 chars): {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}")
        
        print(f"[DEBUG] Endpoint URL: {endpoint}")
        
        match = re.search(r"/deployments/([^/]+)/?$", endpoint)
        if not match:
            error_msg = "Invalid endpoint URL. Expected format: .../deployments/model-name"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        self.model_name = match.group(1)
        
        print(f"[DEBUG] Extracted model name: {self.model_name}")
        print(f"[DEBUG] Timeout set to: {self.timeout} seconds")
        
        try:
            hostname = re.match(r'https?://([^/]+)', endpoint)
            if hostname:
                print(f"[DEBUG] Testing network connectivity to {hostname.group(1)}")
                socket.gethostbyname(hostname.group(1))
                print(f"[DEBUG] Network connectivity to {hostname.group(1)} confirmed")
            else:
                print("[WARNING] Could not extract hostname from endpoint URL")
        except Exception as e:
            print(f"[WARNING] Network connectivity test failed: {str(e)}")
        
        try:
            print("[DEBUG] Creating ChatCompletionsClient")
            self.client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
            print("[DEBUG] ChatCompletionsClient created successfully")
            
            print("[DEBUG] Initializing usage statistics")
            usage_summary = {
                self.model_name: {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                }
            }
            self.client.total_usage_summary = usage_summary
            self.client.actual_usage_summary = usage_summary.copy()
            
            print("[DEBUG] OpenAI strategy initialized successfully")
            print(f"[DEBUG] Python version: {sys.version}")
            print(f"[DEBUG] Using endpoint: {self.endpoint}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChatCompletionsClient: {str(e)}")
            print(traceback.format_exc())
            raise

    def create_agent(self, **kwargs) -> Agent:
        """Create an OpenAI agent.

        Args:
            **kwargs: Arguments to pass to the agent constructor.

        Returns:
            Agent: The created OpenAI agent instance.
        """
        print("[AGENT] Creating CustomAgent")
        
        # Create a custom agent that uses Azure AI SDK
        class CustomAgent(Agent):
            def __init__(self, client: ChatCompletionsClient, model_name: str, verbose: bool = False, timeout: int = 60, **kwargs):
                print(f"[AGENT] Initializing CustomAgent with model: {model_name}")
                super().__init__(**kwargs)
                self.client = client
                self.model_name = model_name
                self.verbose = verbose
                self.timeout = timeout
                self._reply_func = None
                self._reply_func_args = None
                self._reply_func_list = []
                self._reply_func_list_args = []
                self.client_cache = {}
                self.previous_cache = {}
                self.conversation_cache = {}
                self.reply_func_list = []
                self.reply_func_list_args = []
                self.messages = []
                self._last_message = {"content": ""}
                
                print(f"[AGENT] CustomAgent initialized with model: {model_name}")
                print(f"[AGENT] Using timeout: {self.timeout} seconds")

            def register_reply(
                self,
                trigger: Union[type, List[type], None],
                reply_func: Optional[Callable] = None,
                config: Optional[Dict] = None,
            ):
                """Register reply function and configuration.

                Args:
                    trigger: The trigger class or list of trigger classes.
                    reply_func: The reply function to register.
                    config: Optional configuration dictionary.
                """
                print(f"[AGENT] Registering reply function: {reply_func}")
                self.reply_func_list.append(reply_func)
                self.reply_func_list_args.append((trigger, config))

            def last_message(self, sender=None) -> Dict:
                """Get the last message sent by this agent.

                Args:
                    sender: The sender to get the last message for.

                Returns:
                    The last message sent by this agent.
                """
                print(f"[AGENT] Getting last message: {self._last_message}")
                return self._last_message

            def _raise_exception_on_async_reply_functions(self):
                """Check if any reply functions are async."""
                pass

            def can_reply(self, message: Dict, sender: Agent) -> bool:
                """Check if the agent can reply to the message."""
                return True

            def receive(
                self,
                message: Union[Dict, str],
                sender: Agent,
                request_reply: bool = True,
                silent: bool = False,
            ) -> Tuple[bool, Optional[str]]:
                """Receive a message from another agent.

                Args:
                    message: Message received.
                    sender: Sender of the message.
                    request_reply: Whether to request a reply from this agent.
                    silent: Whether to print the message.

                Returns:
                    A tuple of (reply_generated, reply_message).
                """
                print(f"[AGENT] Received message from {sender.name}")
                
                if not silent and not sender.silent:
                    if isinstance(message, dict):
                        content = message.get("content", "")
                    else:
                        content = message
                    print(f"{sender.name}: {content}")

                # Add the message to the history
                if isinstance(message, dict):
                    content = message.get("content", "")
                    self._last_message = message
                else:
                    content = message
                    self._last_message = {"content": content}
                
                print(f"[AGENT] Adding message to history: {content[:50]}{'...' if len(content) > 50 else ''}")
                self.messages.append({"role": "user", "content": content})
                
                print(f"[AGENT] Current message history count: {len(self.messages)}")
                
                if request_reply:
                    print("[AGENT] Generating real response from Azure OpenAI API")
                    
                    try:
                        azure_messages = []
                        for msg in self.messages:
                            if msg.get("role") == "system":
                                azure_messages.append(SystemMessage(content=msg["content"]))
                            else:
                                azure_messages.append(UserMessage(content=msg["content"]))
                        
                        print(f"[AGENT] Converted {len(azure_messages)} messages for API request")
                        
                        response = self.client.complete(
                            messages=azure_messages,
                            max_tokens=4096,
                            temperature=0.7,
                            top_p=0.95,
                            model=self.model_name,
                        )
                        
                        reply = response.choices[0].message.content
                        print(f"[AGENT] Received API response: {reply[:100]}...")
                        
                        self.messages.append({"role": "assistant", "content": reply})
                        self._last_message = {"content": reply}
                        
                        for i, (reply_func, (trigger, config)) in enumerate(zip(self.reply_func_list, self.reply_func_list_args)):
                            if reply_func:
                                print(f"[AGENT] Calling reply function #{i}: {reply_func}")
                                reply_func(self.name, [{"content": reply}], sender, config or {})
                        
                        return True, reply
                        
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        print(f"[ERROR] {error_message}")
                        import traceback
                        print(traceback.format_exc())
                        
                        # Return a fallback response in case of error
                        fallback_reply = "I encountered an error while processing your request. Please check the logs for more details."
                        self.messages.append({"role": "assistant", "content": fallback_reply})
                        self._last_message = {"content": fallback_reply}
                        return True, fallback_reply
                
                return False, None

            def _prepare_chat(
                self,
                recipient: Agent,
                clear_history: bool = True,
                prepare_recipient: bool = True,
                reply_at_receive: bool = True,
            ):
                """Prepare for a chat with a recipient."""
                print(f"[AGENT] Preparing chat with {recipient.name}")
                print(f"[AGENT] clear_history={clear_history}, prepare_recipient={prepare_recipient}, reply_at_receive={reply_at_receive}")
                
                if clear_history:
                    self.clear_history()
                if prepare_recipient and recipient is not None:
                    recipient._prepare_chat(self, clear_history, False, reply_at_receive)

            def clear_history(self):
                """Clear the conversation history."""
                print("[AGENT] Clearing conversation history")
                self.messages.clear()
                self.conversation_cache.clear()
                self._last_message = {"content": ""}

            def generate_response(self, messages, sender, **kwargs):
                print("[API] Starting response generation")
                print(f"[API] Message count: {len(messages)}")
                
                # Convert the message history to Azure AI SDK format
                azure_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        print(f"[API] Adding system message: {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")
                        azure_messages.append(SystemMessage(content=msg["content"]))
                    else:
                        print(f"[API] Adding user message: {msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}")
                        azure_messages.append(UserMessage(content=msg["content"]))
                
                print(f"[API] Converted message count: {len(azure_messages)}")
                print(f"[API] Message types: {[type(m).__name__ for m in azure_messages]}")
                
                print(f"[API] Sending request to Azure OpenAI with model: {self.model_name}")
                print(f"[API] Client endpoint: {self.client._endpoint}")
                print(f"[API] API Key: {self.client._credential.key[:5]}...{self.client._credential.key[-5:] if len(self.client._credential.key) > 10 else ''}")
                
                try:
                    print(f"[API] Beginning API call with timeout {self.timeout} seconds")
                    
                    start_time = time.time()
                    response = None
                    attempts = 0
                    
                    # Perform the API call with timeout
                    while time.time() - start_time < self.timeout:
                        attempts += 1
                        try:
                            print(f"[API] Attempt #{attempts} to call Azure OpenAI API")
                            
                            response = self.client.complete(
                                messages=azure_messages,
                                max_tokens=4096,
                                temperature=0.7,
                                top_p=0.95,
                                model=self.model_name,  # Use model name from endpoint
                            )
                            print("[API] API call completed successfully")
                            break  # Break the loop if successful
                        except ClientAuthenticationError as auth_err:
                            print(f"[ERROR] Authentication error: {str(auth_err)}")
                            print("[ERROR] Please check your API key and endpoint")
                            raise  # No point retrying authentication errors
                        except ServiceRequestError as e:
                            print(f"[WARNING] ServiceRequestError on attempt #{attempts}: {str(e)}")
                            print("[WARNING] Retrying in 3 seconds...")
                            time.sleep(3)  # Wait 3 seconds before retrying
                        except Exception as e:
                            print(f"[ERROR] Unexpected error on attempt #{attempts}: {str(e)}")
                            print(traceback.format_exc())
                            time.sleep(2)  # Wait before retrying
                    
                    if response is None:
                        error_msg = f"API call timed out after {self.timeout} seconds and {attempts} attempts"
                        print(f"[ERROR] {error_msg}")
                        raise TimeoutError(error_msg)
                        
                    print("[API] Processing API response")
                    
                    # Update usage statistics
                    tokens_used = 500  # Estimate as Azure AI SDK doesn't provide this info directly
                    self.client.total_usage_summary[self.model_name]["total_tokens"] += tokens_used
                    self.client.total_usage_summary[self.model_name]["completion_tokens"] += tokens_used // 2
                    self.client.total_usage_summary[self.model_name]["prompt_tokens"] += tokens_used // 2
                    
                    # Update actual usage summary too
                    self.client.actual_usage_summary[self.model_name]["total_tokens"] += tokens_used
                    self.client.actual_usage_summary[self.model_name]["completion_tokens"] += tokens_used // 2
                    self.client.actual_usage_summary[self.model_name]["prompt_tokens"] += tokens_used // 2
                    
                    content = response.choices[0].message.content
                    print(f"[API] Received response content: {content[:50]}{'...' if len(content) > 50 else ''}")
                    
                    for i, (reply_func, (trigger, config)) in enumerate(zip(self.reply_func_list, self.reply_func_list_args)):
                        if reply_func:
                            print(f"[API] Calling reply function #{i}: {reply_func}")
                            reply_func(self.name, [{"content": content}], sender, config or {})
                    
                    return content
                except Exception as e:
                    error_message = f"API Error: {str(e)}\n{traceback.format_exc()}"
                    print(f"[ERROR] {error_message}")
                    
                    # Attempt to get more details about the endpoint and credential
                    try:
                        print("[DEBUG] Attempting to inspect client for additional information")
                        import inspect
                        client_info = inspect.getmembers(self.client)[:10]
                        print(f"[DEBUG] Client info: {client_info}")
                    except Exception as inspect_error:
                        print(f"[DEBUG] Inspection error: {inspect_error}")
                    
                    return f"Error calling Azure OpenAI API: {str(e)}"

        return CustomAgent(client=self.client, model_name=self.model_name, verbose=self.verbose, timeout=self.timeout, **kwargs)

    def get_config_list(self) -> List[Dict[str, Any]]:
        """Get the configuration list for OpenAI.

        Returns:
            List[Dict[str, Any]]: The configuration list for OpenAI.
        """
        print(f"[CONFIG] Returning config list with model: {self.model_name}")
        return [{"endpoint": self.endpoint, "api_key": self.api_key, "model": self.model_name}] 

    def test_connection(self) -> bool:
        """Test the connection to the Azure OpenAI API.
        
        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        try:
            print(f"[TEST] Testing connection to {self.endpoint}")
            
            # Simple completion request to test connection
            response = self.client.complete(
                messages=[UserMessage(content="Tell me hello")],
                max_tokens=10,
                temperature=0.7,
                top_p=0.95,
                model=self.model_name,
            )
            
            # Check if we got a valid response
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                print(f"[TEST] Connection test successful")
                print(f"[TEST] Response: {response.choices[0].message.content}")
                return True
            else:
                print(f"[TEST] Connection test failed: Empty response")
                return False
                
        except ClientAuthenticationError as e:
            print(f"[TEST] Authentication error: {str(e)}")
            return False
        except ServiceRequestError as e:
            print(f"[TEST] Service request error: {str(e)}")
            return False
        except Exception as e:
            print(f"[TEST] Unexpected error: {str(e)}")
            traceback.print_exc()
            return False 