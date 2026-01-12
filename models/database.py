"""
Database Module for Redline.AI
SQLite-based storage for reviews, conversations, and training data.
"""

import sqlite3
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Database location
DB_DIR = Path(__file__).parent.parent / 'data'
DB_PATH = DB_DIR / 'redlineai.db'

# Ensure data directory exists
DB_DIR.mkdir(exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Reviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            project_name TEXT,
            project_number TEXT,
            planset_filename TEXT,
            review_type TEXT,
            initial_report TEXT,
            eval_data TEXT,
            page_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Conversation turns table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id TEXT PRIMARY KEY,
            review_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            content_summary TEXT,
            pages_analyzed TEXT,
            items_referenced TEXT,
            eval_updates TEXT,
            used_vision INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (review_id) REFERENCES reviews(id),
            UNIQUE(review_id, turn_number, role)
        )
    ''')
    
    # Conversation training table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_training (
            id TEXT PRIMARY KEY,
            review_id TEXT,
            turn_id TEXT,
            example_type TEXT,
            user_query TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            context_summary TEXT,
            items_involved TEXT,
            pages_involved TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (review_id) REFERENCES reviews(id),
            FOREIGN KEY (turn_id) REFERENCES conversation_turns(id)
        )
    ''')
    
    # Indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_reviews_session ON reviews(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_turns_review ON conversation_turns(review_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_type ON conversation_training(example_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at)')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


class Review:
    """Review model for database operations."""
    
    @staticmethod
    def create(
        session_id: str,
        project_name: str = None,
        project_number: str = None,
        planset_filename: str = None,
        review_type: str = None,
        initial_report: str = None,
        eval_data: dict = None,
        page_count: int = None
    ) -> str:
        """Create a new review record. Returns review ID."""
        review_id = str(uuid.uuid4())
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reviews (id, session_id, project_name, project_number, 
                               planset_filename, review_type, initial_report, 
                               eval_data, page_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            review_id,
            session_id,
            project_name,
            project_number,
            planset_filename,
            review_type,
            initial_report,
            json.dumps(eval_data) if eval_data else None,
            page_count
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Created review {review_id} for session {session_id}")
        return review_id
    
    @staticmethod
    def get(review_id: str) -> Optional[Dict]:
        """Get a review by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM reviews WHERE id = ?', (review_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = dict(row)
            if data.get('eval_data'):
                data['eval_data'] = json.loads(data['eval_data'])
            return data
        return None
    
    @staticmethod
    def get_by_session(session_id: str, limit: int = 20) -> List[Dict]:
        """Get reviews for a session."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM reviews 
            WHERE session_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            data = dict(row)
            if data.get('eval_data'):
                data['eval_data'] = json.loads(data['eval_data'])
            results.append(data)
        return results
    
    @staticmethod
    def update(review_id: str, **kwargs) -> bool:
        """Update a review record."""
        if not kwargs:
            return False
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Handle eval_data JSON serialization
        if 'eval_data' in kwargs and kwargs['eval_data'] is not None:
            kwargs['eval_data'] = json.dumps(kwargs['eval_data'])
        
        # Build update query
        set_clause = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        set_clause += ", updated_at = ?"
        values = list(kwargs.values()) + [datetime.now().isoformat(), review_id]
        
        cursor.execute(f'''
            UPDATE reviews SET {set_clause} WHERE id = ?
        ''', values)
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    
    @staticmethod
    def get_turn_count(review_id: str) -> int:
        """Get the number of conversation turns for a review."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM conversation_turns 
            WHERE review_id = ? AND role = 'user'
        ''', (review_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count


class ConversationTurn:
    """Conversation turn model for database operations."""
    
    @staticmethod
    def create(
        review_id: str,
        turn_number: int,
        role: str,
        content: str,
        content_summary: str = None,
        pages_analyzed: list = None,
        items_referenced: list = None,
        eval_updates: dict = None,
        used_vision: bool = False
    ) -> str:
        """Create a new conversation turn. Returns turn ID."""
        turn_id = str(uuid.uuid4())
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_turns (id, review_id, turn_number, role, 
                                          content, content_summary, pages_analyzed,
                                          items_referenced, eval_updates, used_vision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            turn_id,
            review_id,
            turn_number,
            role,
            content,
            content_summary,
            json.dumps(pages_analyzed) if pages_analyzed else None,
            json.dumps(items_referenced) if items_referenced else None,
            json.dumps(eval_updates) if eval_updates else None,
            1 if used_vision else 0
        ))
        
        conn.commit()
        conn.close()
        return turn_id
    
    @staticmethod
    def get_for_review(review_id: str) -> List[Dict]:
        """Get all conversation turns for a review."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversation_turns 
            WHERE review_id = ? 
            ORDER BY turn_number, role
        ''', (review_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            data = dict(row)
            for field in ['pages_analyzed', 'items_referenced', 'eval_updates']:
                if data.get(field):
                    data[field] = json.loads(data[field])
            data['used_vision'] = bool(data.get('used_vision'))
            results.append(data)
        return results
    
    @staticmethod
    def get_recent(review_id: str, n_turns: int = 2) -> List[Dict]:
        """Get the most recent n conversation turn pairs."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the highest turn numbers
        cursor.execute('''
            SELECT DISTINCT turn_number FROM conversation_turns 
            WHERE review_id = ? 
            ORDER BY turn_number DESC 
            LIMIT ?
        ''', (review_id, n_turns))
        
        turn_numbers = [row[0] for row in cursor.fetchall()]
        
        if not turn_numbers:
            conn.close()
            return []
        
        # Get those turns
        placeholders = ','.join(['?' for _ in turn_numbers])
        cursor.execute(f'''
            SELECT * FROM conversation_turns 
            WHERE review_id = ? AND turn_number IN ({placeholders})
            ORDER BY turn_number, role
        ''', [review_id] + turn_numbers)
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            data = dict(row)
            for field in ['pages_analyzed', 'items_referenced', 'eval_updates']:
                if data.get(field):
                    data[field] = json.loads(data[field])
            data['used_vision'] = bool(data.get('used_vision'))
            results.append(data)
        return results
    
    @staticmethod
    def update_summary(turn_id: str, summary: str) -> bool:
        """Update the content summary for a turn."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversation_turns SET content_summary = ? WHERE id = ?
        ''', (summary, turn_id))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0


class ConversationTraining:
    """Conversation training example model."""
    
    @staticmethod
    def create(
        review_id: str,
        turn_id: str,
        example_type: str,
        user_query: str,
        ai_response: str,
        context_summary: str = None,
        items_involved: list = None,
        pages_involved: list = None,
        outcome: str = None
    ) -> str:
        """Create a new training example from conversation."""
        example_id = str(uuid.uuid4())
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_training (id, review_id, turn_id, example_type,
                                             user_query, ai_response, context_summary,
                                             items_involved, pages_involved, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            example_id,
            review_id,
            turn_id,
            example_type,
            user_query,
            ai_response,
            context_summary,
            json.dumps(items_involved) if items_involved else None,
            json.dumps(pages_involved) if pages_involved else None,
            outcome
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Created conversation training example {example_id}")
        return example_id
    
    @staticmethod
    def get_by_type(example_type: str = None, limit: int = 100) -> List[Dict]:
        """Get training examples, optionally filtered by type."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if example_type:
            cursor.execute('''
                SELECT * FROM conversation_training 
                WHERE example_type = ?
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (example_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM conversation_training 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            data = dict(row)
            for field in ['items_involved', 'pages_involved']:
                if data.get(field):
                    data[field] = json.loads(data[field])
            results.append(data)
        return results
    
    @staticmethod
    def get_statistics() -> Dict:
        """Get statistics about conversation training data."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total count
        cursor.execute('SELECT COUNT(*) FROM conversation_training')
        total = cursor.fetchone()[0]
        
        # By type
        cursor.execute('''
            SELECT example_type, COUNT(*) as count 
            FROM conversation_training 
            GROUP BY example_type
        ''')
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By outcome
        cursor.execute('''
            SELECT outcome, COUNT(*) as count 
            FROM conversation_training 
            WHERE outcome IS NOT NULL
            GROUP BY outcome
        ''')
        by_outcome = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent examples count (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM conversation_training 
            WHERE created_at > datetime('now', '-7 days')
        ''')
        recent = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_examples': total,
            'by_type': by_type,
            'by_outcome': by_outcome,
            'recent_7_days': recent
        }
    
    @staticmethod
    def get_all_for_export(limit: int = 1000) -> List[Dict]:
        """Get all training examples for export."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ct.*, r.project_name, r.review_type 
            FROM conversation_training ct
            LEFT JOIN reviews r ON ct.review_id = r.id
            ORDER BY ct.created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            data = dict(row)
            for field in ['items_involved', 'pages_involved']:
                if data.get(field):
                    data[field] = json.loads(data[field])
            results.append(data)
        return results


# Initialize database on module import
init_db()
