
# Funes Improvement Plan

This document outlines a development plan to improve the Funes codebase. The plan addresses issues related to code redundancy, architectural inconsistencies, and potential bugs.

## 1. Consolidate Embedding Management

**Issue:** The `EmbeddingManager` class in `memory_manager.py` and the `RAGSystem` class in `rag_system.py` both independently initialize and use a `SentenceTransformer` model. This is redundant and inefficient.

**Plan:**
- **Centralize Embedding Model:** The `EmbeddingManager` will be the single source for the embedding model.
- **Refactor `RAGSystem`:** The `RAGSystem` will receive an `EmbeddingManager` instance (or the model itself) during initialization, removing its own model loading logic.
- **Update `memory_manager.py`:** Ensure the `DualMemoryManager` correctly initializes and passes the `EmbeddingManager` to the `RAGSystem`.

## 2. Refactor `database.py`

**Issue:** The `DatabaseManager` class in `database.py` has a `__del__` method for closing the database connection. This is not a reliable way to manage connections, as `__del__` is not guaranteed to be called when the object goes out of scope.

**Plan:**
- **Implement Context Manager:** Refactor `DatabaseManager` to use a context manager (`__enter__` and `__exit__`) to ensure that the database connection is properly opened and closed.
- **Use `with` statements:** Update all usages of `DatabaseManager` to use `with` statements, ensuring that the connection is managed safely and efficiently.

## 3. Simplify `llm_handler.py`

**Issue:** The `LLMHandler` in `llm_handler.py` has complex logic for handling tools, including specific checks for `specific_tool`. This can be simplified by relying on the tool registration and discovery mechanism in the `tools` package.

**Plan:**
- **Remove `specific_tool` Logic:** The `generate_response` method will no longer need the `specific_tool` parameter. The decision of which tool to use should be handled by the tool selection mechanism.
- **Standardize Tool Handling:** Rely on the `tools.get_tool()` function for all tool lookups, which already includes logic for resolving tool names.

## 4. Remove Redundant Code in `llm_utilities.py`

**Issue:** The `llm_utilities.py` file contains several functions that are either redundant or could be better placed in other modules. For example, `should_use_tools_vector` is tightly coupled with the `DatabaseManager` and `EmbeddingManager`.

**Plan:**
- **Move `should_use_tools_vector`:** This function will be moved to the `memory_manager.py` module, as it is closely related to memory and context management.
- **Deprecate `extract_tool_information`:** The `tools` package already provides functions for getting tool descriptions and parameters. This function is redundant.
- **Review Other Utilities:** Analyze other functions in `llm_utilities.py` to determine if they can be moved to more appropriate modules or removed.

## 5. Improve `rag_system.py`

**Issue:** The `RAGSystem` class in `rag_system.py` initializes its own `DualMemoryManager`, which is unnecessary and creates a circular dependency.

**Plan:**
- **Remove `DualMemoryManager` Initialization:** The `RAGSystem` should not initialize a `DualMemoryManager`. Instead, it should receive the necessary components (like the `DatabaseManager`) during initialization.
- **Streamline `process_file`:** The `process_file` method should directly use the `DatabaseManager` to store the processed file chunks, without going through the `DualMemoryManager`.

## 6. Clean Up `funes-course` Directory

**Issue:** The `funes-course` directory contains educational materials that are not part of the core Funes application. While useful for learning, they add clutter to the main codebase.

**Plan:**
- **Move to a Separate Repository:** The `funes-course` directory will be moved to a separate Git repository. This will keep the main Funes repository focused on the application code.
- **Update `README.md`:** The main `README.md` will be updated to link to the new course repository.

## 7. General Code Quality Improvements

- **Add Type Hinting:** Add type hints to all function signatures to improve code clarity and enable static analysis.
- **Improve Logging:** Ensure that all log messages are informative and consistent. Remove any unnecessary or redundant logging.
- **Add Docstrings:** Add docstrings to all public functions and classes, explaining their purpose, arguments, and return values.
