import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Connect to the MySQL database."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "document_storage")
    )


def save_document(user_id, filename, file_type, file_data):
    """Save an uploaded document and its metadata in MySQL."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        query = """
            INSERT INTO uploaded_documents
                (user_id, filename, file_type, file_data)
            VALUES (%s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (user_id, filename, file_type, file_data)
        )

        connection.commit()
        return cursor.lastrowid

    finally:
        cursor.close()
        connection.close()


def get_user_documents(user_id):
    """Retrieve the metadata for a user's uploaded documents."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT document_id, filename, file_type, uploaded_at
            FROM uploaded_documents
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (user_id,)
        )

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()


def get_document_data(user_id, document_id):
    """Retrieve a document's binary data, checking its owner."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT file_data
            FROM uploaded_documents
            WHERE user_id = %s AND document_id = %s
            """,
            (user_id, document_id)
        )

        result = cursor.fetchone()
        return result[0] if result else None

    finally:
        cursor.close()
        connection.close()

