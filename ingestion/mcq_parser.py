# ingest/mcq_parser.py
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class MCQOption:
    """Represents a single MCQ option"""
    id: str  # A, B, C, D
    text: str
    is_correct: bool

@dataclass
class MCQuestion:
    """Represents a complete MCQ with metadata"""
    question_id: str
    question_text: str
    options: List[MCQOption]
    correct_answer: str
    explanation: Optional[str]
    subject: str
    chapter: str
    topic: str
    difficulty: str
    question_type: str  # factual, conceptual, application, analytical
    distractors: List[str]  # Common wrong answers
    embedding: Optional[np.ndarray] = None

class MCQParser:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        
        # Pattern matching for different MCQ formats
        self.mcq_patterns = {
            'standard': r'(\d+)\.\s*(.*?)\n\s*(?:\([aA]\)|\(A\)|A\.)\s*(.*?)\n\s*(?:\([bB]\)|\(B\)|B\.)\s*(.*?)\n\s*(?:\([cC]\)|\(C\)|C\.)\s*(.*?)\n\s*(?:\([dD]\)|\(D\)|D\.)\s*(.*?)\n',
            'numbered': r'Q(\d+)\.\s*(.*?)\n\s*1\.\s*(.*?)\n\s*2\.\s*(.*?)\n\s*3\.\s*(.*?)\n\s*4\.\s*(.*?)\n',
            'bulleted': r'(\d+)\.\s*(.*?)\n\s*•\s*(.*?)\n\s*•\s*(.*?)\n\s*•\s*(.*?)\n\s*•\s*(.*?)\n'
        }
    
    def parse_text_file(self, file_path: str) -> List[MCQuestion]:
        """Parse MCQs from a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self._extract_mcqs_from_text(content)
    
    def parse_csv_file(self, file_path: str) -> List[MCQuestion]:
        """Parse MCQs from CSV file"""
        df = pd.read_csv(file_path)
        mcqs = []
        
        for idx, row in df.iterrows():
            try:
                mcq = self._create_mcq_from_row(row, idx)
                if mcq:
                    mcqs.append(mcq)
            except Exception as e:
                self.logger.warning(f"Error parsing row {idx}: {e}")
                continue
        
        return mcqs
    
    def parse_json_file(self, file_path: str) -> List[MCQuestion]:
        """Parse MCQs from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mcqs = []
        if isinstance(data, list):
            for item in data:
                mcq = self._create_mcq_from_dict(item)
                if mcq:
                    mcqs.append(mcq)
        
        return mcqs
    
    def _extract_mcqs_from_text(self, text: str) -> List[MCQuestion]:
        """Extract MCQs from raw text using pattern matching"""
        mcqs = []
        
        # Try different patterns
        for pattern_name, pattern in self.mcq_patterns.items():
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    mcq = self._create_mcq_from_match(match, pattern_name)
                    if mcq:
                        mcqs.append(mcq)
                except Exception as e:
                    self.logger.warning(f"Error creating MCQ from match: {e}")
                    continue
        
        return mcqs
    
    def _create_mcq_from_match(self, match: Tuple, pattern_name: str) -> Optional[MCQuestion]:
        """Create MCQ object from regex match"""
        if pattern_name == 'standard' and len(match) >= 6:
            q_num, question, opt_a, opt_b, opt_c, opt_d = match[:6]
            
            options = [
                MCQOption('A', opt_a.strip(), False),
                MCQOption('B', opt_b.strip(), False),
                MCQOption('C', opt_c.strip(), False),
                MCQOption('D', opt_d.strip(), False)
            ]
            
            return MCQuestion(
                question_id=f"Q{q_num}",
                question_text=question.strip(),
                options=options,
                correct_answer="",  # Will be filled later
                explanation=None,
                subject="Unknown",
                chapter="Unknown",
                topic="Unknown",
                difficulty="Medium",
                question_type="Unknown",
                distractors=[]
            )
        
        return None
    
    def _create_mcq_from_row(self, row: pd.Series, idx: int) -> Optional[MCQuestion]:
        """Create MCQ from pandas row"""
        try:
            # Map common column names
            question_col = self._find_column(row, ['question', 'question_text', 'q', 'problem'])
            if not question_col:
                return None
            
            options = []
            for opt_letter in ['A', 'B', 'C', 'D']:
                opt_col = self._find_column(row, [opt_letter, f'option_{opt_letter}', f'opt_{opt_letter}'])
                if opt_col:
                    options.append(MCQOption(opt_letter, str(row[opt_col]).strip(), False))
            
            correct_col = self._find_column(row, ['correct', 'answer', 'correct_answer'])
            correct_answer = str(row[correct_col]).strip() if correct_col else ""
            
            # Mark correct option
            for opt in options:
                if opt.id == correct_answer.upper():
                    opt.is_correct = True
            
            return MCQuestion(
                question_id=f"Q{idx+1}",
                question_text=str(row[question_col]).strip(),
                options=options,
                correct_answer=correct_answer.upper(),
                explanation=str(row.get('explanation', '')),
                subject=str(row.get('subject', 'Unknown')),
                chapter=str(row.get('chapter', 'Unknown')),
                topic=str(row.get('topic', 'Unknown')),
                difficulty=str(row.get('difficulty', 'Medium')),
                question_type=str(row.get('type', 'Unknown')),
                distractors=[]
            )
        
        except Exception as e:
            self.logger.error(f"Error creating MCQ from row {idx}: {e}")
            return None
    
    def _find_column(self, row: pd.Series, possible_names: List[str]) -> Optional[str]:
        """Find column name from possible variations"""
        for name in possible_names:
            if name in row.index:
                return name
            # Case insensitive search
            for col in row.index:
                if col.lower() == name.lower():
                    return col
        return None
    
    def _create_mcq_from_dict(self, data: Dict) -> Optional[MCQuestion]:
        """Create MCQ from dictionary"""
        try:
            options = []
            for opt_data in data.get('options', []):
                if isinstance(opt_data, dict):
                    options.append(MCQOption(
                        id=opt_data.get('id', ''),
                        text=opt_data.get('text', ''),
                        is_correct=opt_data.get('is_correct', False)
                    ))
            
            return MCQuestion(
                question_id=data.get('question_id', ''),
                question_text=data.get('question_text', ''),
                options=options,
                correct_answer=data.get('correct_answer', ''),
                explanation=data.get('explanation'),
                subject=data.get('subject', 'Unknown'),
                chapter=data.get('chapter', 'Unknown'),
                topic=data.get('topic', 'Unknown'),
                difficulty=data.get('difficulty', 'Medium'),
                question_type=data.get('question_type', 'Unknown'),
                distractors=data.get('distractors', [])
            )
        
        except Exception as e:
            self.logger.error(f"Error creating MCQ from dict: {e}")
            return None
    
    def analyze_mcqs(self, mcqs: List[MCQuestion]) -> Dict:
        """Analyze MCQ patterns and extract insights"""
        analysis = {
            'total_questions': len(mcqs),
            'subjects': Counter([mcq.subject for mcq in mcqs]),
            'chapters': Counter([mcq.chapter for mcq in mcqs]),
            'difficulty_distribution': Counter([mcq.difficulty for mcq in mcqs]),
            'question_types': Counter([mcq.question_type for mcq in mcqs]),
            'common_distractors': [],
            'answer_distribution': Counter([mcq.correct_answer for mcq in mcqs])
        }
        
        # Analyze common distractors
        all_distractors = []
        for mcq in mcqs:
            wrong_options = [opt.text for opt in mcq.options if not opt.is_correct]
            all_distractors.extend(wrong_options)
        
        analysis['common_distractors'] = Counter(all_distractors).most_common(20)
        
        return analysis
    
    def detect_trick_patterns(self, mcqs: List[MCQuestion]) -> Dict:
        """Detect common trick patterns in MCQs"""
        trick_patterns = {
            'negation_tricks': [],
            'absolute_statements': [],
            'similar_options': [],
            'numerical_traps': []
        }
        
        for mcq in mcqs:
            question_text = mcq.question_text.lower()
            
            # Negation tricks
            if any(word in question_text for word in ['not', 'except', 'least', 'never']):
                trick_patterns['negation_tricks'].append(mcq.question_id)
            
            # Absolute statements
            if any(word in question_text for word in ['always', 'never', 'all', 'none', 'every']):
                trick_patterns['absolute_statements'].append(mcq.question_id)
            
            # Similar options (lexical similarity)
            option_texts = [opt.text for opt in mcq.options]
            for i, opt1 in enumerate(option_texts):
                for j, opt2 in enumerate(option_texts[i+1:], i+1):
                    similarity = self._calculate_text_similarity(opt1, opt2)
                    if similarity > 0.8:  # High similarity threshold
                        trick_patterns['similar_options'].append(mcq.question_id)
                        break
            
            # Numerical traps (options with similar numbers)
            numbers = []
            for opt in mcq.options:
                nums = re.findall(r'\d+\.?\d*', opt.text)
                numbers.extend([float(n) for n in nums])
            
            if len(numbers) >= 2:
                numbers.sort()
                for i in range(len(numbers)-1):
                    if abs(numbers[i+1] - numbers[i]) < 0.1 * numbers[i]:  # Very close numbers
                        trick_patterns['numerical_traps'].append(mcq.question_id)
                        break
        
        return trick_patterns
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def generate_embeddings(self, mcqs: List[MCQuestion]) -> List[MCQuestion]:
        """Generate embeddings for MCQ questions"""
        questions = [mcq.question_text for mcq in mcqs]
        embeddings = self.embedding_model.encode(questions, show_progress_bar=True)
        
        for mcq, embedding in zip(mcqs, embeddings):
            mcq.embedding = embedding
        
        return mcqs
    
    def save_mcqs(self, mcqs: List[MCQuestion], output_path: str):
        """Save MCQs to JSON file"""
        serializable_mcqs = []
        
        for mcq in mcqs:
            mcq_dict = asdict(mcq)
            if mcq_dict['embedding'] is not None:
                mcq_dict['embedding'] = mcq_dict['embedding'].tolist()
            serializable_mcqs.append(mcq_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_mcqs, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(mcqs)} MCQs to {output_path}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = MCQParser()
    
    # Example: Parse from CSV
    # mcqs = parser.parse_csv_file("mcqs.csv")
    
    # Example: Parse from text file
    # mcqs = parser.parse_text_file("mcqs.txt")
    
    # Analyze MCQs
    # analysis = parser.analyze_mcqs(mcqs)
    # print("MCQ Analysis:", analysis)
    
    # Detect trick patterns
    # tricks = parser.detect_trick_patterns(mcqs)
    # print("Trick Patterns:", tricks)
    
    # Generate embeddings
    # mcqs = parser.generate_embeddings(mcqs)
    
    # Save processed MCQs
    # parser.save_mcqs(mcqs, "processed_mcqs.json")
    
    print("MCQ Parser ready for use!")
