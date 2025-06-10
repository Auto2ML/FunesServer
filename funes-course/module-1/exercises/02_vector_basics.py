#!/usr/bin/env python3
"""
Funes Course - Module 1, Exercise 2: Vector Embeddings Basics
Understanding how text becomes vectors and enables semantic search
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from collections import defaultdict
import math

class SimpleWordEmbedder:
    """
    A simplified word embedding system to demonstrate concepts
    """
    
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.word_to_vector = {}
        self.vocabulary = set()
        
        # Pre-create some example embeddings for common words
        self._create_sample_embeddings()
    
    def _create_sample_embeddings(self):
        """
        Create sample embeddings that demonstrate semantic relationships
        """
        # Seed for reproducible results
        np.random.seed(42)
        
        # Create semantic clusters
        semantic_groups = {
            'animals': ['cat', 'dog', 'bird', 'fish', 'rabbit'],
            'food': ['apple', 'bread', 'cheese', 'pizza', 'cake'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'calm'],
            'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy']
        }
        
        # Create base vectors for each semantic group
        group_centers = {}
        for i, group in enumerate(semantic_groups.keys()):
            # Create a base vector for this semantic group
            center = np.random.randn(self.embedding_dim)
            center = center / np.linalg.norm(center)  # Normalize
            group_centers[group] = center
        
        # Create word embeddings around group centers
        for group, words in semantic_groups.items():
            center = group_centers[group]
            for word in words:
                # Add some noise to the center to create individual word vectors
                noise = np.random.randn(self.embedding_dim) * 0.3
                vector = center + noise
                vector = vector / np.linalg.norm(vector)  # Normalize
                
                self.word_to_vector[word] = vector
                self.vocabulary.add(word)
    
    def get_embedding(self, word):
        """
        Get the embedding vector for a word
        """
        if word.lower() in self.word_to_vector:
            return self.word_to_vector[word.lower()]
        else:
            # Return a random vector for unknown words
            vector = np.random.randn(self.embedding_dim)
            return vector / np.linalg.norm(vector)
    
    def find_similar_words(self, target_word, top_k=5):
        """
        Find words most similar to the target word
        """
        if target_word.lower() not in self.word_to_vector:
            return []
        
        target_vector = self.word_to_vector[target_word.lower()]
        similarities = []
        
        for word, vector in self.word_to_vector.items():
            if word != target_word.lower():
                similarity = np.dot(target_vector, vector)
                similarities.append((word, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class TextEmbeddingSystem:
    """
    A system that converts sentences to embeddings using word embeddings
    """
    
    def __init__(self, word_embedder):
        self.word_embedder = word_embedder
    
    def sentence_to_vector(self, sentence):
        """
        Convert a sentence to a vector by averaging word vectors
        """
        words = sentence.lower().split()
        word_vectors = []
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                vector = self.word_embedder.get_embedding(clean_word)
                word_vectors.append(vector)
        
        if not word_vectors:
            return np.zeros(self.word_embedder.embedding_dim)
        
        # Average the word vectors
        sentence_vector = np.mean(word_vectors, axis=0)
        return sentence_vector / np.linalg.norm(sentence_vector)
    
    def find_similar_sentences(self, target_sentence, sentence_list, top_k=3):
        """
        Find sentences most similar to the target sentence
        """
        target_vector = self.sentence_to_vector(target_sentence)
        similarities = []
        
        for sentence in sentence_list:
            if sentence != target_sentence:
                sentence_vector = self.sentence_to_vector(sentence)
                similarity = np.dot(target_vector, sentence_vector)
                similarities.append((sentence, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def demonstrate_word_similarities():
    """
    Demonstrate how word embeddings capture semantic relationships
    """
    print("🔤 WORD EMBEDDING DEMONSTRATION")
    print("=" * 50)
    
    embedder = SimpleWordEmbedder()
    
    # Test words from different semantic groups
    test_words = ['cat', 'apple', 'red', 'happy', 'sunny']
    
    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar_words = embedder.find_similar_words(word)
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")
    
    print("\n" + "=" * 50)
    print("OBSERVATION: Words with semantic relationships have higher similarity scores!")

def demonstrate_sentence_embeddings():
    """
    Demonstrate how sentence embeddings work for semantic search
    """
    print("\n📝 SENTENCE EMBEDDING DEMONSTRATION")
    print("=" * 50)
    
    embedder = SimpleWordEmbedder()
    text_system = TextEmbeddingSystem(embedder)
    
    # Sample "memories" that might be stored in Funes
    memory_bank = [
        "I love cats and dogs as pets",
        "My favorite food is pizza and cake",
        "Red and blue are beautiful colors",
        "I feel happy when it's sunny outside",
        "Rainy weather makes me feel calm",
        "Birds can fly in the sky",
        "Cheese and bread make a good snack",
        "Green is the color of nature"
    ]
    
    # Test queries
    test_queries = [
        "Tell me about animals",
        "What do you like to eat?",
        "How do you feel about weather?",
        "What colors do you prefer?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("Most relevant memories:")
        
        similar_sentences = text_system.find_similar_sentences(query, memory_bank)
        for i, (sentence, similarity) in enumerate(similar_sentences, 1):
            print(f"  {i}. {sentence} (similarity: {similarity:.3f})")
    
    print("\n" + "=" * 50)
    print("OBSERVATION: The system finds semantically related memories, not just keyword matches!")

def visualize_word_clusters():
    """
    Create a 2D visualization of word embeddings to show clustering
    """
    print("\n📊 WORD CLUSTERING VISUALIZATION")
    print("=" * 50)
    
    embedder = SimpleWordEmbedder()
    
    # Get all words and their embeddings
    words = list(embedder.word_to_vector.keys())
    embeddings = [embedder.word_to_vector[word] for word in words]
    
    # Reduce dimensionality to 2D using PCA-like approach (simplified)
    embeddings_matrix = np.array(embeddings)
    
    # Simple 2D projection (take first 2 dimensions)
    x_coords = embeddings_matrix[:, 0]
    y_coords = embeddings_matrix[:, 1]
    
    # Create color mapping for semantic groups
    semantic_groups = {
        'animals': ['cat', 'dog', 'bird', 'fish', 'rabbit'],
        'food': ['apple', 'bread', 'cheese', 'pizza', 'cake'],
        'colors': ['red', 'blue', 'green', 'yellow', 'purple'],
        'emotions': ['happy', 'sad', 'angry', 'excited', 'calm'],
        'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy']
    }
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    word_colors = {}
    
    for i, (group, group_words) in enumerate(semantic_groups.items()):
        for word in group_words:
            word_colors[word] = colors[i]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for i, word in enumerate(words):
        color = word_colors.get(word, 'gray')
        plt.scatter(x_coords[i], y_coords[i], c=color, s=100, alpha=0.7)
        plt.annotate(word, (x_coords[i], y_coords[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.title('Word Embeddings in 2D Space\n(Colors represent semantic groups)')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.grid(True, alpha=0.3)
    
    # Create legend
    legend_elements = []
    for i, group in enumerate(semantic_groups.keys()):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=10, 
                                        label=group.capitalize()))
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    print("Notice how words from the same semantic group cluster together!")

def similarity_heatmap():
    """
    Create a heatmap showing similarity between different words
    """
    print("\n🔥 SIMILARITY HEATMAP")
    print("=" * 50)
    
    embedder = SimpleWordEmbedder()
    
    # Select a subset of words for the heatmap
    selected_words = ['cat', 'dog', 'apple', 'pizza', 'red', 'blue', 'happy', 'sad']
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(selected_words), len(selected_words)))
    
    for i, word1 in enumerate(selected_words):
        for j, word2 in enumerate(selected_words):
            vec1 = embedder.get_embedding(word1)
            vec2 = embedder.get_embedding(word2)
            similarity = np.dot(vec1, vec2)
            similarity_matrix[i, j] = similarity
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=selected_words, 
                yticklabels=selected_words,
                annot=True, 
                fmt='.3f', 
                cmap='coolwarm',
                center=0)
    
    plt.title('Word Similarity Heatmap\n(Higher values = more similar)')
    plt.tight_layout()
    plt.show()
    
    print("Dark red = high similarity, Dark blue = low similarity")

def funes_memory_simulation():
    """
    Simulate how Funes would use embeddings for memory retrieval
    """
    print("\n🧠 FUNES MEMORY SIMULATION")
    print("=" * 50)
    
    embedder = SimpleWordEmbedder()
    text_system = TextEmbeddingSystem(embedder)
    
    # Simulate Funes' memory bank
    funes_memories = [
        "User mentioned they have a cat named Whiskers",
        "User prefers Italian food like pizza and pasta",
        "User feels happy during sunny weather",
        "User's favorite color is blue",
        "User is learning about machine learning",
        "User works as a software developer",
        "User likes to read science fiction books",
        "User exercises by running in the park",
        "User drinks coffee every morning",
        "User is planning a vacation to Japan"
    ]
    
    # Simulate user queries
    user_queries = [
        "Tell me about my pet",
        "What kind of food do I like?",
        "How do I feel about weather?",
        "What's my job?",
        "What are my hobbies?"
    ]
    
    print("FUNES MEMORY RETRIEVAL SIMULATION")
    print("-" * 40)
    
    for query in user_queries:
        print(f"\nUser Query: '{query}'")
        print("Funes retrieves relevant memories:")
        
        similar_memories = text_system.find_similar_sentences(query, funes_memories)
        for i, (memory, similarity) in enumerate(similar_memories, 1):
            print(f"  {i}. {memory}")
            print(f"     (Relevance: {similarity:.3f})")
    
    print("\n" + "=" * 50)
    print("This is how Funes provides contextual responses using vector similarity!")

def main():
    """
    Main function to run all demonstrations
    """
    print("🧠 FUNES COURSE - Module 1, Exercise 2")
    print("=" * 60)
    print("Understanding Vector Embeddings for Semantic Memory")
    print()
    
    # Run demonstrations
    demonstrate_word_similarities()
    demonstrate_sentence_embeddings()
    
    # Ask if user wants to see visualizations
    print("\n" + "=" * 60)
    show_viz = input("Would you like to see visualizations? (requires matplotlib) (y/n): ").lower().strip()
    if show_viz == 'y':
        try:
            visualize_word_clusters()
            similarity_heatmap()
        except ImportError:
            print("Visualization libraries not available. Skipping plots.")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    # Always run the Funes simulation
    funes_memory_simulation()
    
    print("\n🎉 Exercise 2 Complete!")
    print("\nKey Learning Points:")
    print("• Words with similar meanings have similar vector representations")
    print("• Sentence embeddings enable semantic search (not just keyword matching)")
    print("• Vector similarity measures how 'related' two pieces of text are")
    print("• Funes uses this for intelligent memory retrieval")
    print("\nNext: Run exercise 03_architecture.py")

if __name__ == "__main__":
    main()
