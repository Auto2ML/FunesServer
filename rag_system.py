import os
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from database import DatabaseManager
from memory_manager import DualMemoryManager

class RAGSystem:
    def __init__(self, db_params):
        self.db_manager = DatabaseManager(db_params)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_manager = DualMemoryManager()
        

    def _convert_to_text(self, file_path):
        """Convert documents to markdown using docling"""
        #if not self._is_supported_file(file_path):
        #    raise ValueError(f"Unsupported file type. Supported types are: {self.supported_types}")
        
        converter = DocumentConverter()
        result = converter.convert(file_path)        
        return result.document.export_to_markdown()

    def process_file(self, file_path):
        """Process the uploaded file, split into chunks, and store in the database"""
        max_chunk_size = 512  # Define a reasonable chunk size for the LLM context

        try:
            content = self._convert_to_text(file_path)
            # Split content into chunks
            chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk)
                self.db_manager.insert_memory(chunk, embedding, source=os.path.basename(file_path))
                
        except Exception as e:
            raise RuntimeError(f"Error processing file {file_path}: {str(e)}")
