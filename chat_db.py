import sqlite3
import json
from datetime import datetime

def init_db():
    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            history TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_session(history_data):
    if not history_data:
        return
    conn = sqlite3.connect("chats.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_json = json.dumps(history_data, indent=2)
    cursor.execute("INSERT INTO chat_sessions (timestamp, history) VALUES (?, ?)", (timestamp, history_json))
    conn.commit()
    conn.close()
