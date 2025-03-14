import gradio as gr
from memory_manager import DualMemoryManager
from rag_system import RAGSystem

# Create memory manager instance
memory_manager = DualMemoryManager()

# Initialize RAG system
rag_system = RAGSystem(memory_manager.db_params)

def chat_interface(message, user_history, llm_history):
    """Gradio chat interface handler"""
    # Get LLM response
    response = memory_manager.process_chat(message)
    
    # Update histories
    #user_history = user_history + [{'role': 'user', 'content': message}]
    #llm_history = llm_history + [{'role': 'assistant', 'content': response}]
    
    user_history.append(message)
    llm_history.append(response)
    
    return message, user_history[-1], llm_history[-1]

def delete_memories_by_source_interface(source):
    """Delete memories by source interface"""
    if source:
        memory_manager.db_manager.delete_memories_by_source(source)
        return f"Memories related to '{source}' deleted successfully."
    return "No source selected."

def clear_memories_interface():
    """Gradio memory clearing interface"""
    return memory_manager.clear_memories()

def upload_file_interface(file):
    """Handle file upload and processing"""
    if file is not None:
        rag_system.process_file(file.name)
        return "File processed successfully."
    return "No file uploaded."

def setup_gradio():
    """Set up and return the Gradio Blocks interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Funes: LLM with Dual Memory")
        
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
                source_dropdown = gr.Dropdown(
                    label="Select Source to Delete",
                    choices=memory_manager.db_manager.get_unique_sources()
                )
                delete_source_btn = gr.Button("Delete Selected Source Memories")
                delete_source_output = gr.Textbox(label="Delete Source Memories Result")
                clear_all_output = gr.Textbox(label="Clear All Memories Result")
            
        
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
        
        file_upload.change(
            upload_file_interface,
            inputs=file_upload,
            outputs=file_upload_output
        )

    return demo
