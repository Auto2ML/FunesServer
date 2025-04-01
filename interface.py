import gradio as gr
from config import LLAMAINDEX_CONFIG, DB_CONFIG

# Import the LlamaIndex implementations
from llamaindex_llm import LlamaIndexLLMHandler
from llamaindex_rag import LlamaIndexRAGSystem
from llamaindex_tools import get_all_tools_as_llamaindex

# Create memory manager instance
llm_handler = LlamaIndexLLMHandler()
rag_system = LlamaIndexRAGSystem(DB_CONFIG)

# Track conversation history
chat_history = []

def chat_interface(message, user_history, llm_history):
    """Gradio chat interface handler"""
    # Get relevant memories from RAG system
    try:
        memories = rag_system.query(message, top_k=3)
        additional_context = ""
        if memories:
            additional_context = "Relevant past memories:\n"
            for memory in memories:
                additional_context += f"- {memory[0]}\n"
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        additional_context = None
        memories = []
    
    # Get LLM response
    response = llm_handler.generate_response(
        message,
        conversation_history=user_history,
        additional_context=additional_context
    )
    
    # Update history
    user_history.append({"role": "user", "content": message})
    user_history.append({"role": "assistant", "content": response})
    
    # Update display history
    llm_history.append(message)
    llm_history.append(response)
    
    # Store in RAG system
    try:
        rag_system.store_memory(message, source="chat")
        rag_system.store_memory(response, source="chat")
    except Exception as e:
        print(f"Error storing memory: {str(e)}")
    
    return "", user_history[-2]["content"], llm_history[-1]

def delete_memories_by_source_interface(source):
    """Delete memories by source interface"""
    if source:
        # Use our new delete_memories_by_source method
        success = rag_system.delete_memories_by_source(source)
        if success:
            return f"Memories from source '{source}' deleted successfully."
        else:
            return f"Failed to delete memories from source '{source}'."
    return "No source selected."

def clear_memories_interface():
    """Gradio memory clearing interface"""
    # Use our new clear_memories method
    success = rag_system.clear_memories()
    if success:
        return "All memories cleared successfully."
    else:
        return "Failed to clear memories."

def upload_file_interface(file):
    """Handle file upload and processing"""
    if file is not None:
        result = rag_system.process_file(file.name)
        if result:
            return "File processed successfully."
        else:
            return "Error processing file."
    return "No file uploaded."

def list_sources_interface():
    """List all available sources"""
    # Use our new get_unique_sources method
    return rag_system.get_unique_sources()

def setup_gradio():
    """Set up and return the Gradio Blocks interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Funes: LLM with LlamaIndex RAG")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Conversation Interface")
                user_output = gr.Textbox(
                    label="User Message",
                    show_label=True,
                    interactive=False,
                    lines=10
                )
                llm_output = gr.Textbox(
                    label="LLM Response",
                    show_label=True,
                    interactive=False,
                    lines=10
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Enter your message or question",
                        placeholder="Type your message or question here...",
                        interactive=True
                    )
                    submit_btn = gr.Button("Send")
            
            with gr.Column(scale=1):
                gr.Markdown("## Memory Management")
                
                # File upload widget
                file_upload = gr.File(label="Upload a file", type="filepath")
                file_upload_output = gr.Textbox(label="File Upload Result")
                
                # Clear all memories button
                clear_all_btn = gr.Button("Clear All Memories")
                # Dynamic source dropdown using get_unique_sources
                source_dropdown = gr.Dropdown(
                    label="Select Source to Delete",
                    choices=list_sources_interface(),
                    interactive=True
                )
                delete_source_btn = gr.Button("Delete Selected Source Memories")
                delete_source_output = gr.Textbox(label="Delete Source Memories Result")
                clear_all_output = gr.Textbox(label="Clear All Memories Result")
                
                # Add a refresh button to update the source list
                refresh_sources_btn = gr.Button("Refresh Source List")
        
        # Keep track of conversation for LlamaIndex
        user_history = gr.State([])
        llm_history = gr.State([])
        
        # Message handling
        submit_btn.click(
            chat_interface,
            inputs=[msg, user_history, llm_history],
            outputs=[msg, user_output, llm_output]
        )
        msg.submit(  # Ensure the message box is cleared immediately
            chat_interface,
            inputs=[msg, user_history, llm_history],
            outputs=[msg, user_output, llm_output]
        )
        
        # Clear all memories handling
        clear_all_btn.click(
            clear_memories_interface,
            outputs=clear_all_output
        )
        delete_source_btn.click(
            delete_memories_by_source_interface,
            inputs=source_dropdown,
            outputs=delete_source_output
        )
        
        # File upload handling
        file_upload.change(
            upload_file_interface,
            inputs=file_upload,
            outputs=file_upload_output
        )
        
        # Refresh source list
        refresh_sources_btn.click(
            list_sources_interface,
            outputs=source_dropdown
        )

    return demo
