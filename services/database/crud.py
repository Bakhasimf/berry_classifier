from services.database.connection import get_connection
import psycopg2


def save_prediction(filename, image_bytes, prediction):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (filename, image_data, prediction)
        VALUES (%s, %s, %s)
    """, (filename, psycopg2.Binary(image_bytes), prediction))
    conn.commit()
    cur.close()
    conn.close()
