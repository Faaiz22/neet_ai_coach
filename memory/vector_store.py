"""
Vector Store implementation for AI NEET Coach
Supports FAISS and ChromaDB backends with metadata filtering
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Import embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document structure for vector storage"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass

class EmbeddingModel:
    """Handles text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim: {self.dimension})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.model.encode([text], convert_to_numpy=True)[0]

class FAISSVectorStore(VectorStoreBase):
    """FAISS-based vector store implementation"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        self.embedding_model = EmbeddingModel(embedding_model)
        self.dimension = self.embedding_model.dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents: List[Document] = []
        self.doc_metadata: List[Dict] = []
        
        logger.info(f"Initialized FAISS vector store with dimension {self.dimension}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            return
        
        # Extract texts and generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            self.documents.append(doc)
            self.doc_metadata.append(doc.metadata)
        
        logger.info(f"Added {len(documents)} documents to FAISS store")
    
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(k * 2, self.index.ntotal))  # Get more results for filtering
        
        # Apply metadata filters
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc = self.documents[idx]
            
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append((doc, float(score)))
            if len(results) >= k:
                break
        
        return results
    
    def save(self, path: str) -> None:
        """Save the vector store to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save documents and metadata
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "dimension": self.dimension,
                "num_docs": len(self.documents),
                "embedding_model": self.embedding_model.model.model_name if hasattr(self.embedding_model.model, 'model_name') else "unknown"
            }, f, indent=2)
        
        logger.info(f"Saved FAISS vector store to {path}")
    
    def load(self, path: str) -> None:
        """Load the vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load documents
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        # Update metadata list
        self.doc_metadata = [doc.metadata for doc in self.documents]
        
        logger.info(f"Loaded FAISS vector store from {path}")

class ChromaDBVectorStore(VectorStoreBase):
    """ChromaDB-based vector store implementation"""
    
    def __init__(self, collection_name: str = "neet_coach", embedding_model: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        self.embedding_model = EmbeddingModel(embedding_model)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB vector store: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB"""
        if not documents:
            return
        
        # Prepare data for ChromaDB
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        ids = [doc.doc_id or f"doc_{i}_{hash(doc.content)}" for i, doc in enumerate(documents)]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents in ChromaDB"""
        query_embedding = self.embedding_model.encode_single(query).tolist()
        
        # Prepare where clause for filtering
        where_clause = filters if filters else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )
        
        # Convert results to Document objects
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                doc = Document(
                    content=doc_text,
                    metadata=metadata,
                    doc_id=results['ids'][0][i]
                )
                # Convert distance to similarity score (ChromaDB returns distances)
                similarity_score = 1.0 - distance
                search_results.append((doc, similarity_score))
        
        return search_results
    
    def save(self, path: str) -> None:
        """ChromaDB auto-persists, but we can save metadata"""
        os.makedirs(path, exist_ok=True)
        metadata = {
            "collection_name": self.collection_name,
            "count": self.collection.count(),
            "embedding_model": getattr(self.embedding_model.model, 'model_name', 'unknown')
        }
        
        with open(os.path.join(path, "chromadb_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved ChromaDB metadata to {path}")
    
    def load(self, path: str) -> None:
        """ChromaDB auto-loads, just log info if metadata exists"""
        metadata_path = os.path.join(path, "chromadb_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"ChromaDB collection: {metadata.get('collection_name', 'unknown')} with {metadata.get('count', 0)} documents")

class VectorStore:
    """Main VectorStore class that can use different backends"""
    
    def __init__(self, backend: str = "faiss", **kwargs):
        """
        Initialize vector store with specified backend
        
        Args:
            backend: "faiss" or "chromadb"
            **kwargs: Backend-specific arguments
        """
        self.backend = backend.lower()
        
        if self.backend == "faiss":
            self.store = FAISSVectorStore(**kwargs)
        elif self.backend == "chromadb":
            self.store = ChromaDBVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Choose 'faiss' or 'chromadb'")
        
        logger.info(f"Initialized VectorStore with {self.backend} backend")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self.store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        return self.store.similarity_search(query, k, filters)
    
    def add_ncert_content(self, content_chunks: List[Dict]) -> None:
        """Add NCERT content chunks to vector store"""
        documents = []
        for chunk in content_chunks:
            doc = Document(
                content=chunk['text'],
                metadata={
                    'source': 'ncert',
                    'subject': chunk.get('subject', 'unknown'),
                    'class': chunk.get('class', 'unknown'),
                    'chapter': chunk.get('chapter', 'unknown'),
                    'section': chunk.get('section', 'unknown'),
                    'page': chunk.get('page', 'unknown')
                },
                doc_id=f"ncert_{chunk.get('subject', 'unknown')}_{chunk.get('chapter', 'unknown')}_{hash(chunk['text'])}"
            )
            documents.append(doc)
        
        self.add_documents(documents)
        logger.info(f"Added {len(documents)} NCERT content chunks")
    
    def add_mcq_data(self, mcq_data: List[Dict]) -> None:
        """Add MCQ data to vector store"""
        documents = []
        for mcq in mcq_data:
            # Add question as document
            question_text = f"Question: {mcq.get('question', '')}\nOptions: {', '.join(mcq.get('options', []))}"
            
            doc = Document(
                content=question_text,
                metadata={
                    'source': 'mcq',
                    'subject': mcq.get('subject', 'unknown'),
                    'topic': mcq.get('topic', 'unknown'),
                    'difficulty': mcq.get('difficulty', 'medium'),
                    'correct_answer': mcq.get('correct_answer', ''),
                    'explanation': mcq.get('explanation', ''),
                    'tags': mcq.get('tags', [])
                },
                doc_id=f"mcq_{mcq.get('subject', 'unknown')}_{hash(mcq.get('question', ''))}"
            )
            documents.append(doc)
        
        self.add_documents(documents)
        logger.info(f"Added {len(documents)} MCQ documents")
    
    def search_by_subject(self, query: str, subject: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search within a specific subject"""
        return self.similarity_search(query, k, filters={'subject': subject})
    
    def search_ncert_only(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search only in NCERT content"""
        return self.similarity_search(query, k, filters={'source': 'ncert'})
    
    def search_mcq_only(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search only in MCQ data"""
        return self.similarity_search(query, k, filters={'source': 'mcq'})
    
    def save(self, path: str) -> None:
        """Save the vector store"""
        self.store.save(path)
    
    def load(self, path: str) -> None:
        """Load the vector store"""
        self.store.load(path)

# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vs = VectorStore(backend="faiss")  # or "chromadb"
    
    # Test with sample documents
    sample_docs = [
        Document(
            content="Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
            metadata={'source': 'ncert', 'subject': 'physics', 'class': '11', 'chapter': 'laws_of_motion'}
        ),
        Document(
            content="The mitochondria is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
            metadata={'source': 'ncert', 'subject': 'biology', 'class': '11', 'chapter': 'cell_structure'}
        )
    ]
    
    vs.add_documents(sample_docs)
    
    # Test search
    results = vs.similarity_search("What is Newton's law?", k=2)
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {doc.content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
