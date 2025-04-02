graph TD
    A[User enters question in Gradio interface] --> B[interface.py: chat_interface() receives message]
    B --> C[memory_manager.py: process_chat() processes message]

    C --> D1[Short-term memory handling]
    D1 --> D1a[_clean_short_term() removes expired messages]
    D1a --> D1b[_add_to_short_term() adds user message]

    C --> D2[Long-term memory handling]
    D2 --> D2a[retrieve_relevant_memories() gets related memories]
    D2a --> D2b[llm_handler.get_single_embedding() generates query embedding]
    D2b --> D2c[db_manager.retrieve_memories() searches vector DB]
    
    C --> E[llm_handler.py: generate_response() builds LLM request]
    E --> E1[Build system message with context]
    E1 --> E2[Add conversation history]
    E2 --> E3[Add user input]
    E3 --> E4[Add tool descriptions to prompt if enabled]
    
    E4 --> F[Backend selection based on LLM_CONFIG]
    F --> F1[OllamaBackend]
    F --> F2[LlamaCppBackend]
    F --> F3[HuggingFaceBackend]
    F --> F4[LlamafileBackend]
    
    F1 --> G[Selected backend.generate() sends request to LLM]
    F2 --> G
    F3 --> G
    F4 --> G
    
    G --> H{Does response contain tool calls?}
    
    H -->|Yes| I1[Execute tool calls]
    I1 --> I2[tools module: execute_tool_call() runs the requested tool]
    I2 --> I3[Add tool results to conversation]
    I3 --> I4[Generate final response incorporating tool results]
    I4 --> J[Format response]
    
    H -->|No| J
    
    J --> K[Store conversation in memories]
    K --> K1[store_memory() saves user message]
    K1 --> K2[store_memory() saves LLM response]
    
    K2 --> L[Return response to chat_interface()]
    L --> M[Update UI state with response]
    M --> N[User sees formatted response in Gradio interface]
    
    subgraph "Pre-processing"
    D1
    D1a
    D1b
    end
    
    subgraph "Memory & Context Retrieval"
    D2
    D2a
    D2b
    D2c
    end
    
    subgraph "LLM Request Formation"
    E
    E1
    E2
    E3
    E4
    end
    
    subgraph "LLM Interaction"
    F
    F1
    F2
    F3
    F4
    G
    end
    
    subgraph "Tool