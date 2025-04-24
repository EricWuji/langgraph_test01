import psycopg2
from config.settings import (
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, 
    POSTGRES_PASSWORD, POSTGRES_PORT
)

def check_db_structure():
    """Check the database structure and vector dimensions"""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )
        
        cur = conn.cursor()
        
        # Check if pgvector extension is installed
        cur.execute("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector';")
        pgvector_version = cur.fetchone()
        
        print(f"pgvector extension: {'Installed' if pgvector_version else 'Not installed'}")
        
        # Check if health_records table exists
        cur.execute("SELECT to_regclass('public.health_records');")
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("Health records table does not exist.")
            return
        
        print("Health records table exists.")
        
        # Check table structure
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'health_records';")
        columns = cur.fetchall()
        
        print("\nTable structure:")
        for col_name, col_type in columns:
            print(f"- {col_name}: {col_type}")
        
        # Check vector dimension
        cur.execute("""
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE c.relname = 'health_records' 
            AND a.attname = 'embedding'
            AND t.typname = 'vector';
        """)
        
        result = cur.fetchone()
        if result:
            dimension = result[0]
            print(f"\nVector dimension: {dimension}")
        
        # Check row count
        cur.execute("SELECT COUNT(*) FROM health_records;")
        row_count = cur.fetchone()[0]
        print(f"Number of records: {row_count}")
        
        # If rows exist, show a sample
        if row_count > 0:
            cur.execute("SELECT id, content, metadata FROM health_records LIMIT 1;")
            sample = cur.fetchone()
            print("\nSample record:")
            print(f"ID: {sample[0]}")
            print(f"Content: {sample[1][:100]}...")
            print(f"Metadata: {sample[2]}")
        
    except Exception as e:
        print(f"Error checking database: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_db_structure()