import sqlite3
from datetime import datetime

def save_feedback_job(user_id, model_prediction, confidence, true_label, features):
    conn = sqlite3.connect("feedback_log.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_id TEXT,
            model_prediction TEXT,
            confidence REAL,
            true_label TEXT,
            features TEXT
        )
    ''')

    cursor.execute('''
        INSERT INTO feedback (timestamp, user_id, model_prediction, confidence, true_label, features)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_id,
        model_prediction,
        confidence,
        true_label,
        str(features)
    ))

    conn.commit()
    conn.close()
