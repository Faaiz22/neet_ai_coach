# ingest/ncert_loader.py
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2
import fitz  # PyMuPDF for better PDF handling
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    source: str
    chapter: str
    section: str
    page_num: int
    chunk_id: str
    embedding: Optional[np.ndarray] = None

class NCERTLoader:
    def __init__(self, data_dir: str = "data/ncert_books", chunk_size: int = 512, overlap: int = 50):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page and structure information"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract basic structure hints
            blocks = page.get_text("dict")
            headings = self._extract_headings(blocks)
            
            pages_data.append({
                'page_num': page_num + 1,
                'text': text,
                'headings': headings,
                'source': os.path.basename(pdf_path)
            })
            
        doc.close()
        return pages_data
    
    def _extract_headings(self, blocks_dict: Dict) -> List[str]:
        """Extract potential headings based on font size and formatting"""
        headings = []
        
        for block in blocks_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        
                        # Heuristic for headings: larger font or bold text
                        if font_size > 12 or font_flags & 2**4:  # Bold flag
                            if len(text) > 3 and len(text) < 100:
                                headings.append(text)
        
        return headings
    
    def chunk_text(self, text: str, source: str, chapter: str, page_num: int) -> List[TextChunk]:
        """Split text into overlapping chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    source=source,
                    chapter=chapter,
                    section="",  # Could be enhanced with section detection
                    page_num=page_num,
                    chunk_id=f"{source}_{chapter}_{chunk_id}"
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, len(sentences) - 2):]
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                source=source,
                chapter=chapter,
                section="",
                page_num=page_num,
                chunk_id=f"{source}_{chapter}_{chunk_id}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def detect_chapter_structure(self, pages_data: List[Dict]) -> Dict[str, List[int]]:
        """Detect chapter boundaries in the document"""
        chapter_map = {}
        current_chapter = "Introduction"
        
        for page_data in pages_data:
            page_num = page_data['page_num']
            headings = page_data['headings']
            
            # Look for chapter indicators
            for heading in headings:
                if any(keyword in heading.lower() for keyword in ['chapter', 'unit', 'lesson']):
                    current_chapter = heading
                    break
            
            if current_chapter not in chapter_map:
                chapter_map[current_chapter] = []
            chapter_map[current_chapter].append(page_num)
        
        return chapter_map
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Generate embeddings for text chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def load_and_process_book(self, pdf_path: str) -> List[TextChunk]:
        """Main method to load and process a single NCERT book"""
        self.logger.info(f"Processing book: {pdf_path}")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        # Detect chapter structure
        chapter_map = self.detect_chapter_structure(pages_data)
        
        # Create chunks
        all_chunks = []
        for chapter, page_nums in chapter_map.items():
            chapter_text = ""
            for page_data in pages_data:
                if page_data['page_num'] in page_nums:
                    chapter_text += page_data['text'] + "\n"
            
            # Chunk the chapter text
            chunks = self.chunk_text(
                chapter_text, 
                os.path.basename(pdf_path), 
                chapter, 
                page_nums[0] if page_nums else 1
            )
            all_chunks.extend(chunks)
        
        # Generate embeddings
        all_chunks = self.generate_embeddings(all_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {pdf_path}")
        return all_chunks
    
    def load_all_books(self) -> Dict[str, List[TextChunk]]:
        """Load and process all NCERT books in the data directory"""
        books_data = {}
        
        for pdf_file in self.data_dir.glob("*.pdf"):
            try:
                chunks = self.load_and_process_book(str(pdf_file))
                books_data[pdf_file.stem] = chunks
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        return books_data
    
    def save_processed_data(self, books_data: Dict[str, List[TextChunk]], output_path: str):
        """Save processed chunks to JSON file"""
        serializable_data = {}
        
        for book_name, chunks in books_data.items():
            serializable_data[book_name] = []
            for chunk in chunks:
                chunk_data = {
                    'content': chunk.content,
                    'source': chunk.source,
                    'chapter': chunk.chapter,
                    'section': chunk.section,
                    'page_num': chunk.page_num,
                    'chunk_id': chunk.chunk_id,
                    'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
                }
                serializable_data[book_name].append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processed data to {output_path}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = NCERTLoader()
    
    # Process all books
    books_data = loader.load_all_books()
    
    # Save processed data
    loader.save_processed_data(books_data, "processed_ncert_data.json")
    
    # Print summary
    for book_name, chunks in books_data.items():
        print(f"{book_name}: {len(chunks)} chunks")
