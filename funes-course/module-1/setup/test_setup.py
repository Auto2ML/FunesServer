#!/usr/bin/env python3
"""
Funes Course - Module 1: Setup Validation
Test script to verify that the development environment is correctly set up
"""

import sys
import importlib
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def test(self, description, test_func):
        """Run a test and track results"""
        try:
            test_func()
            print(f"✓ {description}")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"✗ {description}: {str(e)}")
            self.failures.append((description, str(e)))
            self.tests_failed += 1
            return False
    
    def summary(self):
        """Print test summary"""
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Tests run: {total}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\nFailures:")
            for desc, error in self.failures:
                print(f"  - {desc}: {error}")
        
        if self.tests_failed == 0:
            print(f"\n🎉 All tests passed! Your environment is ready for Module 1.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the setup.")
        
        return self.tests_failed == 0

def test_python_version():
    """Test Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        raise Exception(f"Python {version.major}.{version.minor} detected, need 3.8+")

def test_import_numpy():
    """Test numpy import and basic functionality"""
    import numpy as np
    arr = np.array([1, 2, 3])
    if arr.sum() != 6:
        raise Exception("Numpy basic operations failed")

def test_import_sentence_transformers():
    """Test sentence-transformers import"""
    from sentence_transformers import SentenceTransformer
    # Don't actually load a model in the test, just check import

def test_import_psycopg2():
    """Test psycopg2 import"""
    import psycopg2
    # Just check import, don't test connection

def test_import_matplotlib():
    """Test matplotlib import"""
    import matplotlib.pyplot as plt
    # Just check import

def test_import_dotenv():
    """Test python-dotenv import"""
    from dotenv import load_dotenv

def test_project_structure():
    """Test that basic project structure exists"""
    required_dirs = [
        "funes-project",
        "funes-project/src",
        "funes-project/tests",
        "funes-project/data"
    ]
    
    for directory in required_dirs:
        if not Path(directory).exists():
            raise Exception(f"Directory {directory} not found")

def test_vector_operations():
    """Test basic vector operations needed for embeddings"""
    import numpy as np
    
    # Create sample vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    
    # Test dot product
    dot_product = np.dot(vec1, vec2)
    expected = 32  # 1*4 + 2*5 + 3*6 = 32
    if dot_product != expected:
        raise Exception(f"Dot product failed: {dot_product} != {expected}")
    
    # Test cosine similarity calculation
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = dot_product / (norm1 * norm2)
    
    if not 0.9 < cosine_sim < 1.0:  # Should be high similarity
        raise Exception(f"Cosine similarity calculation seems wrong: {cosine_sim}")

def test_basic_embedding():
    """Test basic embedding functionality (without downloading models)"""
    import numpy as np
    
    # Simulate embedding vectors (384 dimensions like all-MiniLM-L6-v2)
    embedding_dim = 384
    text1_embedding = np.random.rand(embedding_dim)
    text2_embedding = np.random.rand(embedding_dim)
    
    # Test similarity calculation
    similarity = np.dot(text1_embedding, text2_embedding) / (
        np.linalg.norm(text1_embedding) * np.linalg.norm(text2_embedding)
    )
    
    if not -1 <= similarity <= 1:
        raise Exception(f"Similarity out of expected range: {similarity}")

def main():
    print("🧠 Funes Course - Module 1 Setup Validation")
    print("=" * 50)
    
    runner = TestRunner()
    
    # Core Python tests
    runner.test("Python version (3.8+)", test_python_version)
    
    # Package imports
    runner.test("Import numpy", test_import_numpy)
    runner.test("Import sentence-transformers", test_import_sentence_transformers)
    runner.test("Import psycopg2", test_import_psycopg2)
    runner.test("Import matplotlib", test_import_matplotlib)
    runner.test("Import python-dotenv", test_import_dotenv)
    
    # Project structure
    runner.test("Project structure", test_project_structure)
    
    # Functionality tests
    runner.test("Vector operations", test_vector_operations)
    runner.test("Basic embedding simulation", test_basic_embedding)
    
    # Print summary and return success status
    success = runner.summary()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
