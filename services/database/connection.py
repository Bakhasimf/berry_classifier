import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="admin",
        host="db",
        port=5432
    )
