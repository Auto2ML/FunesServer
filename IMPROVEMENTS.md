# Funes Improvement and Implementation Plan

## 1. Enhanced LLM Backend Support

**Suggestion:** The `llm_handler.py` is currently focused on the Ollama backend. To fully deliver on the project's promise of supporting multiple backends, it should be refactored to provide robust and equivalent support for `llama.cpp` and `HuggingFace`.

**Implementation Plan:**

1.  **Abstract Base Class:** Create an abstract base class for LLM handlers that defines a common interface for generating responses and handling tool calls.
2.  **Concrete Implementations:** Implement separate handler classes for Ollama, `llama.cpp`, and HuggingFace, each inheriting from the base class.
3.  **Factory Function:** Create a factory function in `llm_handler.py` that returns the appropriate handler instance based on the `backend_type` specified in `config.py`.
4.  **Update `memory_manager.py`:** Modify the `DualMemoryManager` to use the factory function to get the correct LLM handler.

## 2. Improved Tool Detection and Execution

**Suggestion:** The current tool detection relies on vector similarity, which is a good start. This can be augmented with a more traditional, keyword-based approach to improve accuracy and reduce the chances of the LLM failing to use a tool when it should.

**Implementation Plan:**

1.  **Keyword-based Detection:** In `llm_utilities.py`, create a function that uses regular expressions to detect keywords associated with each tool (e.g., "weather," "time," "date").
2.  **Hybrid Approach:** In `memory_manager.py`, use both the vector-based and keyword-based detection methods. If either one (or both) suggests using a tool, the system should proceed with the tool call.
3.  **Refine Tool Execution:** Consolidate the tool execution logic. The `llm_handler.py` currently has the primary tool execution logic, which is correct. The `memory_manager.py` should only be responsible for detecting *if* a tool should be used and then passing that information to the `llm_handler`.

## 3. Secure and Flexible Configuration

**Suggestion:** The `config.py` file currently contains hardcoded database credentials. This is a security risk and makes the application less flexible. These should be loaded from environment variables, as suggested by the `.env` file mentioned in the `README.md`.

**Implementation Plan:**

1.  **Add `python-dotenv`:** Add the `python-dotenv` library to `requirements.txt`.
2.  **Load Environment Variables:** At the top of `config.py`, use `dotenv.load_dotenv()` to load variables from a `.env` file.
3.  **Update `DB_CONFIG`:** Modify the `DB_CONFIG` dictionary to get its values from `os.environ.get()`, with the current hardcoded values as fallbacks.
4.  **Update `README.md`:** Ensure the `README.md` clearly instructs users to create a `.env` file and specifies the required variables.

## 4. Advanced RAG Chunking Strategy

**Suggestion:** The `rag_system.py` uses a fixed-size chunking strategy. This can be improved by using a more sophisticated method, like recursive character text splitting, which attempts to split text on semantic boundaries (paragraphs, sentences, etc.) first.

**Implementation Plan:**

1.  **Add `langchain`:** Add the `langchain` library to `requirements.txt` for its text splitting utilities.
2.  **Implement Recursive Splitting:** In `rag_system.py`, replace the manual chunking with `langchain.text_splitter.RecursiveCharacterTextSplitter`.
3.  **Configurable Chunking:** Expose the `chunk_size` and `chunk_overlap` parameters in `config.py` so they can be easily tuned.

## 5. Comprehensive Testing Suite

**Suggestion:** The project currently lacks an automated testing suite. Adding tests would significantly improve its reliability and make future development easier and safer.

**Implementation Plan:**

1.  **Add `pytest`:** Add `pytest` to a new `requirements-dev.txt` file.
2.  **Create `tests/` Directory:** Create a `tests/` directory in the project root.
3.  **Unit Tests:**
    *   Write unit tests for the `DatabaseManager` to ensure database operations work as expected (this may require a separate test database).
    *   Write unit tests for the `EmbeddingManager` to verify that embeddings are generated correctly.
    *   Write unit tests for the RAG system's text conversion and chunking logic.
4.  **Integration Tests:**
    *   Write integration tests for the `DualMemoryManager` to simulate a conversation and verify that memory is being stored and retrieved correctly.
    *   Write integration tests for the `llm_handler` to ensure it can communicate with a mock LLM service and handle tool calls.
5.  **CI/CD Pipeline:** As a further step, a GitHub Actions workflow could be created to automatically run the tests on every push and pull request.

## 6. Refactor for Code Quality and Maintainability

**Suggestion:** There are a few areas where the code could be refactored to reduce duplication and improve clarity.

**Implementation Plan:**

1.  **Centralize Logging:** Create a single `get_logger()` function in a `utils.py` file that all other modules can use to get a configured logger instance. This will remove the duplicate logging setup code in `funes.py` and `llm_handler.py`.
2.  **Consolidate Tool Logic:** Review the tool-related functions in `llm_utilities.py` and `llm_handler.py`. Move any overlapping logic into a single, authoritative place. For example, the logic for formatting tool descriptions should be in one place.
