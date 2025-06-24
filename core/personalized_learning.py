# personalized_learning.py
import random
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class MCQQuestion:
    id: str
    subject: str
    topic: str
    subtopic: str
    question: str
    options: List[str]
    correct_answer: str
    difficulty: int  # 1-5 scale
    explanation: str
    concepts: List[str]

class DifficultyScorer:
    def __init__(self):
        self.weights = {
            'concept_complexity': 0.3,
            'calculation_steps': 0.25,
            'prerequisite_knowledge': 0.2,
            'time_required': 0.15,
            'error_prone_factors': 0.1
        }
    
    def score_mcq(self, question_data: Dict) -> int:
        """Score MCQ difficulty from 1-5 based on various factors"""
        score = 0
        
        # Concept complexity (0-5)
        concept_words = ['advanced', 'complex', 'multi-step', 'integration', 'derivation']
        complexity = sum(1 for word in concept_words if word in question_data.get('question', '').lower())
        score += min(complexity, 5) * self.weights['concept_complexity']
        
        # Calculation steps (estimated from question length and formula count)
        text = question_data.get('question', '')
        formula_indicators = ['=', '∫', '∂', '√', '^', 'log', 'sin', 'cos', 'tan']
        calc_complexity = sum(1 for indicator in formula_indicators if indicator in text)
        score += min(calc_complexity, 5) * self.weights['calculation_steps']
        
        # Prerequisite knowledge (based on topic depth)
        topic_depth = {
            'basic': 1, 'intermediate': 3, 'advanced': 5,
            'fundamental': 1, 'application': 3, 'analysis': 5
        }
        prereq_score = 3  # default
        for level, points in topic_depth.items():
            if level in question_data.get('topic', '').lower():
                prereq_score = points
                break
        score += prereq_score * self.weights['prerequisite_knowledge']
        
        # Time required (estimated from question length)
        question_length = len(question_data.get('question', ''))
        time_score = min(question_length / 100, 5)  # normalize to 1-5
        score += time_score * self.weights['time_required']
        
        # Error-prone factors (multiple correct-looking options, tricky wording)
        error_indicators = ['except', 'not', 'incorrect', 'false', 'never']
        error_score = sum(1 for word in error_indicators if word in text.lower())
        score += min(error_score, 5) * self.weights['error_prone_factors']
        
        # Normalize to 1-5 scale
        final_score = max(1, min(5, round(score)))
        return final_score

class PersonalizedLearningLoop:
    def __init__(self, db_path: str = "personalized_learning.db"):
        self.db_path = db_path
        self.difficulty_scorer = DifficultyScorer()
        self.init_database()
    
    def init_database(self):
        """Initialize database for personalized learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                current_level INTEGER DEFAULT 1,
                subjects_preference TEXT, -- JSON
                learning_pace TEXT DEFAULT 'medium', -- slow, medium, fast
                target_score INTEGER DEFAULT 500,
                strong_topics TEXT, -- JSON
                weak_topics TEXT, -- JSON
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                starting_difficulty INTEGER,
                ending_difficulty INTEGER,
                questions_attempted INTEGER,
                accuracy_rate REAL,
                time_spent INTEGER, -- in minutes
                topics_covered TEXT, -- JSON
                performance_trend TEXT -- improving, stable, declining
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_bank (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                question TEXT NOT NULL,
                options TEXT NOT NULL, -- JSON array
                correct_answer TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                explanation TEXT,
                concepts TEXT, -- JSON array
                usage_count INTEGER DEFAULT 0,
                avg_time_taken REAL DEFAULT 0,
                success_rate REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user_profile(self, user_id: str, preferences: Dict = None):
        """Create or update user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        default_prefs = {
            'subjects': ['Physics', 'Chemistry', 'Biology'],
            'difficulty_preference': 'adaptive',
            'session_length': 30  # minutes
        }
        
        if preferences:
            default_prefs.update(preferences)
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, subjects_preference, learning_pace, target_score)
            VALUES (?, ?, ?, ?)
        ''', (user_id, json.dumps(default_prefs), 
              preferences.get('pace', 'medium'),
              preferences.get('target_score', 500)))
        
        conn.commit()
        conn.close()
    
    def get_adaptive_difficulty(self, user_id: str, subject: str, topic: str) -> int:
        """Calculate adaptive difficulty based on user performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent performance for this topic
        cursor.execute('''
            SELECT accuracy_rate, ending_difficulty 
            FROM adaptive_sessions 
            WHERE user_id = ? AND topics_covered LIKE ?
            ORDER BY session_date DESC LIMIT 5
        ''', (user_id, f'%{topic}%'))
        
        recent_sessions = cursor.fetchall()
        conn.close()
        
        if not recent_sessions:
            return 2  # Start with moderate difficulty
        
        # Calculate adaptive difficulty
        avg_accuracy = sum(session[0] for session in recent_sessions) / len(recent_sessions)
        last_difficulty = recent_sessions[0][1]
        
        if avg_accuracy > 80:
            new_difficulty = min(5, last_difficulty + 1)
        elif avg_accuracy < 50:
            new_difficulty = max(1, last_difficulty - 1)
        else:
            new_difficulty = last_difficulty
        
        return new_difficulty
    
    def generate_personalized_session(self, user_id: str, duration_minutes: int = 30,
                                    focus_topics: List[str] = None) -> Dict:
        """Generate a personalized learning session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user profile
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        
        if not profile:
            self.create_user_profile(user_id)
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            profile = cursor.fetchone()
        
        subjects_pref = json.loads(profile[2]) if profile[2] else {'subjects': ['Physics']}
        
        # Get weak topics if focus_topics not provided
        if not focus_topics:
            cursor.execute('''
                SELECT DISTINCT topic FROM adaptive_sessions 
                WHERE user_id = ? AND accuracy_rate < 70
                ORDER BY session_date DESC LIMIT 3
            ''', (user_id,))
            focus_topics = [row[0] for row in cursor.fetchall()]
            if not focus_topics:
                focus_topics = ['Mechanics', 'Thermodynamics', 'Optics']  # Default topics
        
        # Generate session plan
        session_plan = {
            'user_id': user_id,
            'duration_minutes': duration_minutes,
            'focus_topics': focus_topics,
            'questions': [],
            'difficulty_progression': [],
            'estimated_questions': duration_minutes // 2,  # 2 minutes per question
            'session_goals': []
        }
        
        # Generate questions for each topic
        questions_per_topic = max(1, session_plan['estimated_questions'] // len(focus_topics))
        
        for topic in focus_topics:
            adaptive_difficulty = self.get_adaptive_difficulty(user_id, 'Physics', topic)
            
            # Get questions from database
            cursor.execute('''
                SELECT id, question, options, correct_answer, difficulty, explanation
                FROM question_bank 
                WHERE topic = ? AND difficulty BETWEEN ? AND ?
                ORDER BY RANDOM() LIMIT ?
            ''', (topic, max(1, adaptive_difficulty-1), min(5, adaptive_difficulty+1), 
                  questions_per_topic))
            
            questions = cursor.fetchall()
            
            for q in questions:
                session_plan['questions'].append({
                    'id': q[0],
                    'topic': topic,
                    'question': q[1],
                    'options': json.loads(q[2]),
                    'correct_answer': q[3],
                    'difficulty': q[4],
                    'explanation': q[5]
                })
                session_plan['difficulty_progression'].append(q[4])
        
        # Set session goals
        avg_difficulty = np.mean(session_plan['difficulty_progression']) if session_plan['difficulty_progression'] else 3
        target_accuracy = max(60, 100 - (avg_difficulty * 10))
        
        session_plan['session_goals'] = [
            f"Target accuracy: {target_accuracy}%",
            f"Focus on: {', '.join(focus_topics)}",
            f"Average difficulty: {avg_difficulty:.1f}/5"
        ]
        
        conn.close()
        return session_plan
    
    def record_session_result(self, user_id: str, session_data: Dict):
        """Record the results of a learning session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate session statistics
        total_questions = len(session_data.get('responses', []))
        correct_answers = sum(1 for r in session_data.get('responses', []) if r.get('correct', False))
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        starting_difficulty = min(session_data.get('difficulty_progression', [3]))
        ending_difficulty = max(session_data.get('difficulty_progression', [3]))
        
        # Determine performance trend
        recent_accuracy = self._get_recent_accuracy(cursor, user_id)
        if accuracy > recent_accuracy + 10:
            trend = 'improving'
        elif accuracy < recent_accuracy - 10:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Insert session record
        cursor.execute('''
            INSERT INTO adaptive_sessions 
            (user_id, starting_difficulty, ending_difficulty, questions_attempted,
             accuracy_rate, time_spent, topics_covered, performance_trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, starting_difficulty, ending_difficulty, total_questions,
              accuracy, session_data.get('time_spent', 0),
              json.dumps(session_data.get('topics', [])), trend))
        
        conn.commit()
        conn.close()
        
        return {
            'accuracy': round(accuracy, 2),
            'improvement': trend,
            'next_recommended_difficulty': self._calculate_next_difficulty(accuracy, ending_difficulty),
            'weak_areas': self._identify_weak_areas(session_data.get('responses', []))
        }
    
    def _get_recent_accuracy(self, cursor, user_id: str) -> float:
        """Get average accuracy from recent sessions"""
        cursor.execute('''
            SELECT AVG(accuracy_rate) FROM adaptive_sessions 
            WHERE user_id = ? AND session_date >= date('now', '-7 days')
        ''', (user_id,))
        result = cursor.fetchone()[0]
        return result if result else 50.0
    
    def _calculate_next_difficulty(self, accuracy: float, current_difficulty: int) -> int:
        """Calculate recommended difficulty for next session"""
        if accuracy >= 85:
            return min(5, current_difficulty + 1)
        elif accuracy <= 40:
            return max(1, current_difficulty - 1)
        else:
            return current_difficulty
    
    def _identify_weak_areas(self, responses: List[Dict]) -> List[str]:
        """Identify weak areas from session responses"""
        topic_performance = {}
        
        for response in responses:
            topic = response.get('topic', 'Unknown')
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            
            topic_performance[topic]['total'] += 1
            if response.get('correct', False):
                topic_performance[topic]['correct'] += 1
        
        weak_areas = []
        for topic, perf in topic_performance.items():
            accuracy = (perf['correct'] / perf['total']) * 100 if perf['total'] > 0 else 0
            if accuracy < 60:
                weak_areas.append(topic)
        
        return weak_areas
    
    def get_learning_analytics(self, user_id: str, days_back: int = 30) -> Dict:
        """Get comprehensive learning analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_filter = datetime.now() - timedelta(days=days_back)
        
        cursor.execute('''
            SELECT session_date, accuracy_rate, ending_difficulty, 
                   questions_attempted, performance_trend
            FROM adaptive_sessions
            WHERE user_id = ? AND session_date >= ?
            ORDER BY session_date
        ''', (user_id, date_filter))
        
        sessions = cursor.fetchall()
        
        if not sessions:
            return {'message': 'No data available for analysis'}
        
        # Calculate analytics
        accuracies = [s[1] for s in sessions]
        difficulties = [s[2] for s in sessions]
        
        analytics = {
            'total_sessions': len(sessions),
            'average_accuracy': round(np.mean(accuracies), 2),
            'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
            'current_difficulty_level': difficulties[-1],
            'difficulty_progression': round(np.mean(difficulties), 2),
            'total_questions_attempted': sum(s[3] for s in sessions),
            'consistency_score': round(100 - np.std(accuracies), 2),
            'recent_performance': sessions[-5:] if len(sessions) >= 5 else sessions
        }
        
        conn.close()
        return analytics

# Example usage
if __name__ == "__main__":
    learning_loop = PersonalizedLearningLoop()
    
    # Create user profile
    learning_loop.create_user_profile("student123", {
        'subjects': ['Physics', 'Chemistry'],
        'pace': 'medium',
        'target_score': 600
    })
    
    # Generate personalized session
    session = learning_loop.generate_personalized_session("student123", duration_minutes=20)
    print("Session Plan:", json.dumps(session, indent=2))
    
    # Record session results (example)
    sample_responses = [
        {'topic': 'Mechanics', 'correct': True},
        {'topic': 'Mechanics', 'correct': False},
        {'topic': 'Thermodynamics', 'correct': True}
    ]
    
    result = learning_loop.record_session_result("student123", {
        'responses': sample_responses,
        'difficulty_progression': [2, 3, 2],
        'time_spent': 15,
        'topics': ['Mechanics', 'Thermodynamics']
    })
    
    print("Session Result:", json.dumps(result, indent=2))