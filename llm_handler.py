import ollama
from sentence_transformers import SentenceTransformer
from config import LLM_CONFIG, EMBEDDING_CONFIG
import abc
import requests
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import llama_cpp

class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends"""
    
    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the model using the provided messages"""
        pass

class OllamaBackend(LLMBackend):
    """Ollama backend implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Make sure we have a clean messages structure
        formatted_messages = []
        
        # Process and clean each message
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
        
        # Send to Ollama API
        response = ollama.chat(model=self.model_name, messages=formatted_messages)
        
        # Return only the content of the response
        return response['message']['content']

class LlamaCppBackend(LLMBackend):
    """llama.cpp backend implementation"""
    
    def __init__(self, model_path: str, context_size: int = 4096, 
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=context_size,
            temperature=temperature
        )
        self.max_tokens = max_tokens
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Make sure we have clean messages
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
        
        # Convert messages to llama.cpp format
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                if not prompt:
                    prompt += f"<s>[INST] {content} [/INST]"
                else:
                    prompt += f"{content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} </s><s>[INST] "
        
        # Handle case where we might end with an unclosed instruction tag
        if prompt.endswith("[INST] "):
            prompt = prompt[:-7]  # Remove the trailing "[INST] "
        
        response = self.llm(
            prompt=prompt,
            max_tokens=self.max_tokens,
            echo=False
        )
        return response["choices"][0]["text"]

class HuggingFaceBackend(LLMBackend):
    """HuggingFace backend implementation"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Make sure we have clean messages
        formatted_messages = []
        for msg in messages:
            if msg["role"] in ["system", "user", "assistant"]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"].strip()
                })
        
        # Convert messages to a prompt format
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        
        prompt += "<|assistant|>\n"
        
        result = self.pipe(
            prompt,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]
        
        # Extract only the assistant's response
        assistant_response = result.split("<|assistant|>\n")[-1].strip()
        return assistant_response

class LLMHandler:
    def __init__(self, embedding_model=None, llm_model=None, backend_type=None):
        # Initialize embedding model with config or override
        embedding_model_name = embedding_model or EMBEDDING_CONFIG['model_name']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize LLM model with config or override
        llm_model_name = llm_model or LLM_CONFIG['model_name']
        backend_type = backend_type or LLM_CONFIG.get('backend_type', 'ollama')  # Get from config or default to ollama
        
        # Create the appropriate backend
        if backend_type.lower() == "ollama":
            self.llm_backend = OllamaBackend(llm_model_name)
        elif backend_type.lower() == "llamacpp":
            self.llm_backend = LlamaCppBackend(llm_model_name)
        elif backend_type.lower() == "huggingface":
            self.llm_backend = HuggingFaceBackend(llm_model_name)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        # System prompt from config
        self.system_prompt = LLM_CONFIG.get('system_prompt', '')
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.embedding_model.encode(texts).tolist()
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embedding_model.encode(text).tolist()
    
    def generate_response(self, 
                         user_input: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None, 
                         additional_context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM
        
        Args:
            user_input: The user's query
            conversation_history: Optional list of previous conversation messages
            additional_context: Optional context information to add to the system prompt
            
        Returns:
            The generated response from the LLM
        """
        # Initialize messages with system prompt
        messages = []
        
        # Add system message with additional context if provided
        system_message = self.system_prompt
        if additional_context:
            system_message += f"\n\nContext: {additional_context}"
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate response using the backend
        response = self.llm_backend.generate(messages)
        
        return response