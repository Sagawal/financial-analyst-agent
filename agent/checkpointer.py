from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from config.settings import LANGGRAPH_DATABASE_URL


if not LANGGRAPH_DATABASE_URL:
    raise ValueError("LANGGRAPH_DATABASE_URL is missing.")


# Create PostgreSQL connection pool
connection_pool = ConnectionPool(
    conninfo=LANGGRAPH_DATABASE_URL,
    min_size=1,
    max_size=5,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row
    }
)

# Create PostgreSQL checkpointer
checkpointer = PostgresSaver(connection_pool)

# Initialize checkpoint tables
checkpointer.setup()