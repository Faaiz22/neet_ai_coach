# feedback_tracker.py
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import sqlite3

class FeedbackTracker:
    def __init__(self, db_path: str = "neet_feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for tracking feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                question_id TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                user_answer TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                difficulty_level INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weak_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                error_count INTEGER DEFAULT 0,
                total_attempts INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, subject, topic, subtopic)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_response(self, user_id: str, subject: str, topic: str, 
                       question_id: str, correct_answer: str, user_answer: str,
                       subtopic: str = None, difficulty_level: int = 1):
        """Record user's response to a question"""
        is_correct = correct_answer.strip().lower() == user_answer.strip().lower()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert response record
        cursor.execute('''
            INSERT INTO user_responses 
            (user_id, subject, topic, subtopic, question_id, correct_answer, 
             user_answer, is_correct, difficulty_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, subject, topic, subtopic, question_id, 
              correct_answer, user_answer, is_correct, difficulty_level))
        
        # Update weak topics
        self._update_weak_topics(cursor, user_id, subject, topic, subtopic, is_correct)
        
        conn.commit()
        conn.close()
        
        return is_correct
    
    def _update_weak_topics(self, cursor, user_id: str, subject: str, 
                           topic: str, subtopic: str, is_correct: bool):
        """Update weak topics based on user response"""
        cursor.execute('''
            INSERT OR IGNORE INTO weak_topics 
            (user_id, subject, topic, subtopic, error_count, total_attempts)
            VALUES (?, ?, ?, ?, 0, 0)
        ''', (user_id, subject, topic, subtopic))
        
        if is_correct:
            cursor.execute('''
                UPDATE weak_topics 
                SET total_attempts = total_attempts + 1,
                    accuracy_rate = (total_attempts - error_count) * 100.0 / (total_attempts + 1),
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND subject = ? AND topic = ? AND subtopic = ?
            ''', (user_id, subject, topic, subtopic))
        else:
            cursor.execute('''
                UPDATE weak_topics 
                SET error_count = error_count + 1,
                    total_attempts = total_attempts + 1,
                    accuracy_rate = (total_attempts - error_count + 1) * 100.0 / (total_attempts + 1),
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND subject = ? AND topic = ? AND subtopic = ?
            ''', (user_id, subject, topic, subtopic))
    
    def get_weak_topics(self, user_id: str, min_attempts: int = 3, 
                       max_accuracy: float = 60.0) -> List[Dict]:
        """Get user's weak topics based on performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT subject, topic, subtopic, error_count, total_attempts, 
                   accuracy_rate, last_updated
            FROM weak_topics
            WHERE user_id = ? AND total_attempts >= ? AND accuracy_rate <= ?
            ORDER BY accuracy_rate ASC, error_count DESC
        ''', (user_id, min_attempts, max_accuracy))
        
        weak_topics = []
        for row in cursor.fetchall():
            weak_topics.append({
                'subject': row[0],
                'topic': row[1],
                'subtopic': row[2],
                'error_count': row[3],
                'total_attempts': row[4],
                'accuracy_rate': round(row[5], 2),
                'last_updated': row[6]
            })
        
        conn.close()
        return weak_topics
    
    def get_topic_performance(self, user_id: str, subject: str = None,
                             days_back: int = 30) -> Dict:
        """Get detailed performance analysis for topics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_filter = datetime.now() - timedelta(days=days_back)
        
        if subject:
            cursor.execute('''
                SELECT topic, subtopic, is_correct, difficulty_level
                FROM user_responses
                WHERE user_id = ? AND subject = ? AND timestamp >= ?
            ''', (user_id, subject, date_filter))
        else:
            cursor.execute('''
                SELECT topic, subtopic, is_correct, difficulty_level
                FROM user_responses
                WHERE user_id = ? AND timestamp >= ?
            ''', (user_id, date_filter))
        
        performance = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'difficulty_sum': 0
        })
        
        for row in cursor.fetchall():
            topic_key = f"{row[0]}_{row[1] or 'general'}"
            performance[topic_key]['total'] += 1
            performance[topic_key]['difficulty_sum'] += row[3] or 1
            if row[2]:
                performance[topic_key]['correct'] += 1
        
        # Calculate accuracy and average difficulty
        result = {}
        for topic, data in performance.items():
            accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
            avg_difficulty = data['difficulty_sum'] / data['total'] if data['total'] > 0 else 0
            
            result[topic] = {
                'accuracy': round(accuracy, 2),
                'total_questions': data['total'],
                'correct_answers': data['correct'],
                'avg_difficulty': round(avg_difficulty, 2)
            }
        
        conn.close()
        return result
    
    def suggest_practice_topics(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Suggest topics for practice based on weak performance"""
        weak_topics = self.get_weak_topics(user_id)
        
        # Prioritize topics with more errors and lower accuracy
        suggestions = []
        for topic in weak_topics[:limit]:
            priority_score = (100 - topic['accuracy_rate']) + (topic['error_count'] * 5)
            
            suggestions.append({
                'subject': topic['subject'],
                'topic': topic['topic'],
                'subtopic': topic['subtopic'],
                'reason': f"Low accuracy: {topic['accuracy_rate']}%",
                'priority_score': round(priority_score, 2),
                'recommended_questions': max(5, topic['error_count'] * 2)
            })
        
        return sorted(suggestions, key=lambda x: x['priority_score'], reverse=True)
    
    def get_learning_progress(self, user_id: str, days_back: int = 7) -> Dict:
        """Track learning progress over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_filter = datetime.now() - timedelta(days=days_back)
        
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as total,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct
            FROM user_responses
            WHERE user_id = ? AND timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (user_id, date_filter))
        
        progress = {}
        for row in cursor.fetchall():
            date = row[0]
            total = row[1]
            correct = row[2]
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            progress[date] = {
                'total_questions': total,
                'correct_answers': correct,
                'accuracy': round(accuracy, 2)
            }
        
        conn.close()
        return progress

# Example usage
if __name__ == "__main__":
    tracker = FeedbackTracker()
    
    # Record some sample responses
    tracker.record_response("user123", "Physics", "Mechanics", "q1", "A", "B", "Kinematics", 2)
    tracker.record_response("user123", "Physics", "Mechanics", "q2", "C", "C", "Kinematics", 3)
    tracker.record_response("user123", "Chemistry", "Organic", "q3", "B", "A", "Hydrocarbons", 1)
    
    # Get weak topics
    weak = tracker.get_weak_topics("user123")
    print("Weak Topics:", json.dumps(weak, indent=2))
    
    # Get suggestions
    suggestions = tracker.suggest_practice_topics("user123")
    print("Practice Suggestions:", json.dumps(suggestions, indent=2))