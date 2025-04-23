# Database configuration
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "langgraph_db",
    "user": "postgres",
    "password": "postgres"
}

# Build connection string
def get_pg_connection_string() -> str:
    return f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"