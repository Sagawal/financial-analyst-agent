import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import CONVERSATION_DB_PATH

DB_PATH = CONVERSATION_DB_PATH


# Persistent SQLite connection
connection = sqlite3.connect(
    str(DB_PATH),
    check_same_thread=False
)

# Create checkpointer
checkpointer = SqliteSaver(connection)

# Initialize database tables
checkpointer.setup()