import os
from docling.document_converter import DocumentConverter
from database import DatabaseManager
from memory_manager import EmbeddingManager

class RAGSystem:
    def __init__(self, db_manager: DatabaseManager, embedding_manager: EmbeddingManager):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager

    def _convert_to_text(self, file_path):
        """Convert documents to markdown using docling"""
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
                embedding = self.embedding_manager.get_embedding(chunk)
                self.db_manager.insert_memory(chunk, embedding, source=os.path.basename(file_path))
                
        except Exception as e:
            raise RuntimeError(f"Error processing file {file_path}: {str(e)}")
