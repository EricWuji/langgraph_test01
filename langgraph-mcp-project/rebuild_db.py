import os
import psycopg2
from src.utils.ingest import ingest_health_records
from config.settings import (
    POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, 
    POSTGRES_PASSWORD, POSTGRES_PORT
)

def rebuild_database():
    """Drop and rebuild the health records database"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )
        
        conn.autocommit = True
        cur = conn.cursor()
        
        # Drop the existing table
        print("Dropping existing health_records table...")
        cur.execute("DROP TABLE IF EXISTS health_records;")
        print("Table dropped successfully.")
        
        conn.close()
        
        # Re-ingest the health records
        print("\nRe-ingesting health records...")
        success = ingest_health_records()
        
        if success:
            print("\n✅ Database rebuild completed successfully!")
        else:
            print("\n❌ Database rebuild failed.")
        
    except Exception as e:
        print(f"Error rebuilding database: {e}")

if __name__ == "__main__":
    response = input("This will delete all existing health records and rebuild the database. Continue? (y/n): ")
    if response.lower() == 'y':
        rebuild_database()
    else:
        print("Database rebuild cancelled.")