import os
import sqlite3
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "cleaned_books.csv")

DB_PATH = os.path.join(DATA_DIR, "books_database.db")


def create_database():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "cleaned_books.csv not found. Run scraper.py first."
        )

  
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        raise ValueError("The cleaned CSV file contains no records.")


    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

       
        cursor.execute("PRAGMA foreign_keys = ON")


        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            )
        """)

        cursor.execute("DELETE FROM books")
        cursor.execute("DELETE FROM categories")

        categories = sorted(
            df["category"].dropna().unique()
        )

        for category in categories:
            cursor.execute("""
                INSERT INTO categories (category_name)
                VALUES (?)
            """, (category,))

        cursor.execute("""
            SELECT category_id, category_name
            FROM categories
        """)

        category_mapping = {
            name: category_id
            for category_id, name in cursor.fetchall()
        }

     
        for _, row in df.iterrows():

            category_id = category_mapping.get(row["category"])

            if category_id is None:
                continue

            cursor.execute("""
                INSERT INTO books (
                    title,
                    price_gbp,
                    price_inr,
                    rating,
                    in_stock,
                    category_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                category_id
            ))

        connection.commit()

        cursor.execute("SELECT COUNT(*) FROM books")
        book_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM categories")
        category_count = cursor.fetchone()[0]

        print("\nDATABASE CREATED SUCCESSFULLY")
        print("--------------------------------")
        print(f"Database location: {DB_PATH}")
        print(f"Total books inserted: {book_count}")
        print(f"Total categories inserted: {category_count}")

      
        sample = pd.read_sql_query("""
            SELECT
                b.book_id,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.rating,
                b.in_stock,
                c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            LIMIT 5
        """, connection)

        print("\nSAMPLE DATABASE RECORDS")
        print(sample.to_string(index=False))

    finally:
        connection.close()

if __name__ == "__main__":
    create_database()
