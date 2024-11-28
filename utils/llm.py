from typing import Any, List, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import requests

class NvidiaLLM(LLM):
    model: str
    api_key: str
    callback_handler: Any = None
    system_message: str = ""
    tools: List[Any] = Field(default_factory=list)
    
    def bind_tools(self, tools: List[Any]) -> "NvidiaLLM":
        """Bind tools to the LLM and return self for chaining."""
        self.tools = tools
        return self
    
    @property
    def _llm_type(self) -> str:
        return "nvidia_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Construct messages with system message and tools
        messages = []
        
        # Add system message if present
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
            
        # Add tools description if present
        if self.tools:
            tool_descriptions = "\n\n".join([
                f"Tool {i+1}: {tool.name}\n{tool.description}"
                for i, tool in enumerate(self.tools)
            ])
            messages.append({
                "role": "system",
                "content": f"You have access to the following tools:\n\n{tool_descriptions}"
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"https://api.nvcf.nvidia.com/v2/models/{self.model}/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error calling NVIDIA API: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            raise

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "api_key": "**********",  # Mask the API key in logs
            "has_system_message": bool(self.system_message),
            "num_tools": len(self.tools)
        }