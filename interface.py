import gradio as gr
from memory_manager import DualMemoryManager, embedding_manager
from rag_system import RAGSystem
from database import DatabaseManager
from config import DB_CONFIG
import json
import logging
import tools

# Configure logger
logger = logging.getLogger('Interface')

# Create memory manager instance
memory_manager = DualMemoryManager()

# Global variable to store the last accessed memories
last_accessed_memories = []

def chat_interface(message, history):
    """Gradio chat interface handler"""
    global last_accessed_memories
    
    try:
        # Get LLM response
        response = memory_manager.process_chat(message)
        
        # Capture memories that were accessed during this interaction
        try:
            # Get relevant memories for display
            last_accessed_memories = memory_manager.retrieve_relevant_memories(message, top_k=3)
            logger.info(f"Retrieved {len(last_accessed_memories)} memories for display")
        except Exception as e:
            logger.error(f"Error retrieving memories for display: {str(e)}")
            last_accessed_memories = []
        
        # For the Gradio Chatbot component, we need to return the updated history
        # This creates a copy of the history and appends the new response
        history_copy = history.copy()
        # Update the last message that has None as the response
        for i in range(len(history_copy)-1, -1, -1):
            if history_copy[i][1] is None:
                history_copy[i][1] = response
                break
        
        logger.info(f"Chat interface returning updated history with {len(history_copy)} messages")
        return history_copy
    except Exception as e:
        logger.error(f"Error in chat_interface: {str(e)}", exc_info=True)
        # Return error message as the bot's response
        history_copy = history.copy()
        for i in range(len(history_copy)-1, -1, -1):
            if history_copy[i][1] is None:
                history_copy[i][1] = f"Error: {str(e)}"
                break
        return history_copy

def update_memory_display():
    """Update the memory display with the most recently accessed memories"""
    global last_accessed_memories
    
    if not last_accessed_memories:
        return "*No relevant memories were accessed for this query*"
    
    memory_html = "<div class='memory-container'>"
    for i, memory in enumerate(last_accessed_memories):
        memory_text = memory[0]
        # Truncate long memories for display
        if len(memory_text) > 300:
            memory_text = memory_text[:297] + "..."
        
        memory_html += f"<div class='memory-item'><b>Memory {i+1}:</b><br>{memory_text}</div>"
    
    memory_html += "</div>"
    return memory_html

def clear_chat_history():
    """Clear the chat interface history"""
    global last_accessed_memories
    last_accessed_memories = []
    return []

def refresh_sources():
    """Refresh the list of sources in the dropdown"""
    with DatabaseManager(DB_CONFIG) as db_manager:
        return gr.Dropdown(choices=db_manager.get_unique_sources())

def delete_memories_by_source_interface(source):
    """Delete memories by source interface"""
    if source:
        with DatabaseManager(DB_CONFIG) as db_manager:
            db_manager.delete_memories_by_source(source)
        return f"Memories related to '{source}' deleted successfully."
    return "No source selected."

def clear_memories_interface():
    """Gradio memory clearing interface"""
    with DatabaseManager(DB_CONFIG) as db_manager:
        db_manager.clear_memories()
    return "All memories cleared successfully."

def upload_file_interface(file):
    """Handle file upload and processing"""
    if file is not None:
        with DatabaseManager(DB_CONFIG) as db_manager:
            rag_system = RAGSystem(db_manager=db_manager, embedding_manager=embedding_manager)
            rag_system.process_file(file.name)
        return f"File '{file.name}' processed successfully and added to long-term memory."
    return "No file uploaded."

def get_memory_stats():
    """Get basic memory statistics"""
    try:
        with DatabaseManager(DB_CONFIG) as db_manager:
            # Get unique sources
            sources = db_manager.get_unique_sources()
            
            # Count total memories
            memory_count = 0
            source_counts = {}
            
            for source in sources:
                # This would require a new method in DatabaseManager to count memories by source
                # For now, we'll just show the source names
                source_counts[source] = "✓"
                
            return f"**Sources in Memory:** {len(sources)}"
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        return "**Memory Stats:** Error retrieving statistics"

def get_available_tools_info():
    """Get information about all available tools"""
    try:
        available_tools = tools.get_available_tools()
        tool_info_html = "<div class='tools-container'>"
        
        if not available_tools:
            return "<p>No tools available</p>"
        
        for i, tool in enumerate(available_tools):
            if "function" in tool:
                function = tool["function"]
                name = function.get("name", "")
                description = function.get("description", "No description available")
                
                # Extract parameter info
                params_html = ""
                if "parameters" in function and "properties" in function["parameters"]:
                    params_html = "<ul class='tool-params'>"
                    for param_name, param_info in function["parameters"]["properties"].items():
                        param_desc = param_info.get("description", "")
                        required = param_name in function["parameters"].get("required", [])
                        req_label = "<span class='required'>Required</span>" if required else "<span class='optional'>Optional</span>"
                        
                        params_html += f"<li><b>{param_name}</b> {req_label}<br>{param_desc}</li>"
                    params_html += "</ul>"
                
                tool_info_html += f"""
                <div class='tool-item'>
                    <h4>{name}</h4>
                    <p>{description}</p>
                    <details>
                        <summary>Parameters</summary>
                        {params_html}
                    </details>
                </div>
                """
        
        tool_info_html += "</div>"
        return tool_info_html
    except Exception as e:
        logger.error(f"Error getting tool info: {str(e)}")
        return "<p>Error retrieving tool information</p>"

def setup_gradio():
    """Set up and return the Gradio Blocks interface"""
    with gr.Blocks(css="""
        footer {visibility: hidden}
        .memory-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
            background-color: #f9f9f9;
        }
        .memory-item {
            margin-bottom: 12px;
            padding: 8px;
            border-left: 3px solid #2196F3;
            background-color: white;
        }
        .tools-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 16px;
        }
        .tool-item {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 12px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tool-item h4 {
            margin-top: 0;
            color: #2196F3;
        }
        .tool-params {
            padding-left: 20px;
        }
        .required {
            color: #f44336;
            font-size: 0.8em;
            margin-left: 8px;
        }
        .optional {
            color: #4caf50;
            font-size: 0.8em;
            margin-left: 8px;
        }
    """) as demo:
        gr.Markdown("# 🧠 Funes: LLM with Dual Memory")
        
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface with modern chat UI
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        bubble_full_width=False,
                        avatar_images=(None, "🧠"),
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Ask me anything...",
                            show_label=False,
                            scale=9,
                            container=False,
                        )
                        submit_btn = gr.Button("Send", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat History")
                        
                with gr.Column(scale=1, visible=True) as memory_sidebar:
                    memory_stats = gr.Markdown(get_memory_stats())
                    
                    with gr.Accordion("Memory Insights", open=True):
                        gr.Markdown("### Recent Memory Access")
                        memory_display = gr.HTML("*No memories accessed yet*")
            
            # Message handling
            submit_click_event = submit_btn.click(
                lambda user_message, history: (user_message, history + [[user_message, None]]),
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                chat_interface,
                [msg, chatbot],
                chatbot
            ).then(
                update_memory_display,
                None,
                memory_display
            ).then(
                get_memory_stats,
                None,
                memory_stats
            )
            
            msg.submit(
                lambda user_message, history: (user_message, history + [[user_message, None]]),
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            ).then(
                chat_interface,
                [msg, chatbot],
                chatbot
            ).then(
                update_memory_display,
                None,
                memory_display
            ).then(
                get_memory_stats,
                None,
                memory_stats
            )
            
            # Clear chat history
            clear_btn.click(
                clear_chat_history,
                None,
                chatbot
            ).then(
                lambda: "*No memories accessed yet*",
                None,
                memory_display
            )
            
        with gr.Tab("Memory Management"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Upload and process files
                    gr.Markdown("### Document Processing")
                    file_upload = gr.File(label="Upload a document", type="filepath")
                    file_upload_output = gr.Textbox(label="Upload Status", interactive=False)
                    
                    gr.Markdown("""
                    ### Supported File Types
                    - Text (.txt)
                    - Markdown (.md)
                    - PDF (.pdf)
                    - Word (.docx)
                    - HTML (.html)
                    """)
                    
                with gr.Column(scale=1):
                    # Memory management controls
                    gr.Markdown("### Memory Operations")
                    with gr.Row():
                        clear_all_btn = gr.Button("Clear All Memories", variant="stop")
                        clear_all_output = gr.Textbox(label="Operation Result", interactive=False)
                    
                    with gr.Row():
                        source_dropdown = gr.Dropdown(
                            label="Select Source to Delete",
                            choices=memory_manager.get_unique_sources(),
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄 Refresh Sources")
                    
                    delete_source_btn = gr.Button("Delete Selected Source")
                    delete_source_output = gr.Textbox(label="Operation Result", interactive=False)
            
            # Memory visualization placeholder
            with gr.Row():
                gr.Markdown("### Memory Visualization")
                memory_viz = gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <p><i>Memory visualization will be available in a future update.</i></p>
                    <p>The visualization will show memory connections and relevance scores.</p>
                </div>
                """)
        
        with gr.Tab("Tools"):
            gr.Markdown("# Funes Tool System")
            gr.Markdown("""
            Funes uses a variety of tools to enhance its capabilities. These tools allow the system to 
            access real-time information and perform specific tasks that go beyond the knowledge
            in its training data.
            """)
            
            with gr.Row():
                gr.Markdown("## Available Tools")
                tool_info = gr.HTML(get_available_tools_info())
            
            with gr.Row():
                gr.Markdown("## Tool Detection")
                gr.Markdown("""
                Funes automatically determines when to use tools based on:
                1. **Vector-based matching** - Finding tools that are semantically similar to your query
                2. **Keyword detection** - Identifying specific tool-related keywords in your question
                
                For example, asking "What's the weather in Tokyo?" will trigger the weather tool, 
                while "What time is it in Paris?" will use the date/time tool.
                """)
        
        with gr.Tab("About Funes"):
            gr.Markdown("""
            # About Funes
            
            Funes is an enhanced LLM memory system with dual memory and RAG capabilities, inspired by Jorge Luis Borges' short story "Funes the Memorious."
            
            ## Features
            
            - **Dual Memory System**: Short-term conversation memory and long-term persistent memory
            - **Vector-based Retrieval**: Finds semantically similar past memories to provide context
            - **Document Processing**: Upload and process documents to extend the knowledge base
            - **Tool System**: Real-time data access through specialized tools like weather and date/time
            
            ## Memory Architecture
            
            Funes uses a PostgreSQL database with pgvector for storing vector embeddings, allowing for semantic search and retrieval of relevant memories during conversations.
            
            ## Tools
            
            Available tools include:
            - **DateTime Tool**: Provides current date and time for different locations
            - **Weather Tool**: Retrieves weather information for specified locations
            
            *More features coming soon!*
            """)
        
        # File upload handling
        file_upload.change(
            upload_file_interface,
            inputs=file_upload,
            outputs=file_upload_output
        ).then(
            get_memory_stats,
            None,
            memory_stats
        )
        
        # Clear all memories handling
        clear_all_btn.click(
            clear_memories_interface,
            outputs=clear_all_output
        ).then(
            get_memory_stats,
            None,
            memory_stats
        )
        
        # Delete memories by source
        delete_source_btn.click(
            delete_memories_by_source_interface,
            inputs=source_dropdown,
            outputs=delete_source_output
        ).then(
            get_memory_stats,
            None,
            memory_stats
        ).then(
            refresh_sources,
            outputs=source_dropdown
        )
        
        # Refresh sources
        refresh_btn.click(
            refresh_sources,
            outputs=source_dropdown
        )

    return demo
