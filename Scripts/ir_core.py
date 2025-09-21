"""
Core Information Retrieval System
Implements TF-IDF vectorization and cosine similarity search
"""

import os
import re
import math
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import numpy as np

class Document:
    """Represents a document in the IR system"""
    
    def __init__(self, doc_id: str, title: str, content: str, filepath: str = ""):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.filepath = filepath
        self.tokens = self._tokenize(content)
        self.term_freq = Counter(self.tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into tokens and filter out empty strings
        tokens = [token for token in text.split() if token.strip()]
        return tokens
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, title='{self.title[:30]}...')"

class InformationRetrievalSystem:
    """Main IR System with TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.vocabulary: set = set()
        self.idf_scores: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.is_indexed = False
    
    def add_document(self, doc_id: str, title: str, content: str, filepath: str = "") -> None:
        """Add a document to the collection"""
        doc = Document(doc_id, title, content, filepath)
        self.documents[doc_id] = doc
        self.vocabulary.update(doc.tokens)
        self.is_indexed = False  # Need to reindex after adding documents
    
    def load_documents_from_directory(self, directory_path: str) -> int:
        """Load all text files from a directory"""
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist!")
            return 0
        
        loaded_count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        doc_id = filename.replace('.txt', '')
                        title = filename.replace('.txt', '').replace('_', ' ').title()
                        self.add_document(doc_id, title, content, filepath)
                        loaded_count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return loaded_count
    
    def _calculate_idf(self) -> None:
        """Calculate Inverse Document Frequency for all terms"""
        total_docs = len(self.documents)
        
        for term in self.vocabulary:
            # Count documents containing this term
            doc_freq = sum(1 for doc in self.documents.values() if term in doc.term_freq)
            # Calculate IDF with smoothing
            self.idf_scores[term] = math.log(total_docs / (1 + doc_freq))
    
    def _calculate_tf_idf_vectors(self) -> None:
        """Calculate TF-IDF vectors for all documents"""
        for doc_id, doc in self.documents.items():
            vector = {}
            max_freq = max(doc.term_freq.values()) if doc.term_freq else 1
            
            for term in self.vocabulary:
                tf = doc.term_freq.get(term, 0) / max_freq  # Normalized TF
                idf = self.idf_scores[term]
                vector[term] = tf * idf
            
            self.doc_vectors[doc_id] = vector
    
    def build_index(self) -> None:
        """Build the search index (TF-IDF vectors)"""
        if not self.documents:
            print("No documents to index!")
            return
        
        print(f"Building index for {len(self.documents)} documents...")
        self._calculate_idf()
        self._calculate_tf_idf_vectors()
        self.is_indexed = True
        print("Index built successfully!")
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _create_query_vector(self, query: str) -> Dict[str, float]:
        """Create TF-IDF vector for a query"""
        # Tokenize query
        query_doc = Document("query", "Query", query)
        query_tokens = query_doc.tokens
        query_tf = Counter(query_tokens)
        
        # Create query vector
        query_vector = {}
        max_freq = max(query_tf.values()) if query_tf else 1
        
        for term in query_tokens:
            if term in self.vocabulary:
                tf = query_tf[term] / max_freq
                idf = self.idf_scores[term]
                query_vector[term] = tf * idf
        
        return query_vector
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Search for documents matching the query"""
        if not self.is_indexed:
            self.build_index()
        
        if not query.strip():
            return []
        
        # Create query vector
        query_vector = self._create_query_vector(query)
        
        if not query_vector:
            return []
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                similarities.append((doc_id, similarity, self.documents[doc_id].title))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get the full content of a document"""
        if doc_id in self.documents:
            return self.documents[doc_id].content
        return None
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "indexed": self.is_indexed,
            "average_doc_length": np.mean([len(doc.tokens) for doc in self.documents.values()]) if self.documents else 0
        }

def main():
    """Demo function"""
    # Create IR system
    ir_system = InformationRetrievalSystem()
    
    # Add sample documents
    sample_docs = [
        ("doc1", "Python Programming", "Python is a high-level programming language. It's great for data science and web development."),
        ("doc2", "Machine Learning", "Machine learning is a subset of artificial intelligence. It uses algorithms to learn from data."),
        ("doc3", "Data Science", "Data science combines statistics, programming, and domain expertise to extract insights from data."),
        ("doc4", "Web Development", "Web development involves creating websites and web applications using various technologies."),
        ("doc5", "Artificial Intelligence", "AI is the simulation of human intelligence in machines that are programmed to think and learn.")
    ]
    
    for doc_id, title, content in sample_docs:
        ir_system.add_document(doc_id, title, content)
    
    # Build index
    ir_system.build_index()
    
    # Test search
    query = "machine learning algorithms"
    print(f"\nSearching for: '{query}'")
    results = ir_system.search(query, top_k=3)
    
    for i, (doc_id, score, title) in enumerate(results, 1):
        print(f"{i}. {title} (Score: {score:.4f})")
        print(f"   Document ID: {doc_id}")
        print()

if __name__ == "__main__":
    main()
