import ollama
from sentence_transformers import SentenceTransformer
from config import LLM_CONFIG, EMBEDDING_CONFIG

class LLMHandler:
    def __init__(self, embedding_model=None, llm_model=None):
        # Initialize embedding model with config or override
        embedding_model_name = embedding_model or EMBEDDING_CONFIG['model_name']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize LLM model name with config or override
        self.model_name = llm_model or LLM_CONFIG['model_name']
        self.system_prompt = LLM_CONFIG['system_prompt']
    
    def get_embeddings(self, text):
        """Generate embeddings for a text using the embedding model"""
        return self.embedding_model.encode(text)
    
    def generate_response(self, context, user_message):
        """Generate a response from the LLM using context and user message"""
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': f"{context}\n\nCurrent user message: {user_message}"
            }
        ]
        
        try:
            # Call the LLM
            response = ollama.chat(model=self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            error_msg = f"Error generating LLM response: {str(e)}"
            print(error_msg)
            return error_msg
