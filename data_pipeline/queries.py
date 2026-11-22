import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "books_database.db"
)


def run_queries():

    if not os.path.exists(DB_PATH):
        print("Database not found. Run database.py first.")
        return

    connection = sqlite3.connect(DB_PATH)


    connection.execute("PRAGMA foreign_keys = ON")

    queries = {

        "Query 1 - Display all books": """
            SELECT title, price_gbp, rating
            FROM books
        """,

        "Query 2 - Books costing more than 30 GBP": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp > 30
        """,

        "Query 3 - Books sorted by price": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
        """,

        "Query 4 - Five most expensive books": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 5
        """,

        "Query 5 - Unique book categories": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name
        """,

        "Query 6 - Books priced between 10 and 30 GBP": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
        """,

        "Query 7 - Books with their category names (JOIN)": """
            SELECT
                books.title,
                books.price_gbp,
                categories.category_name
            FROM books
            JOIN categories
            ON books.category_id = categories.category_id
        """
    }

    for query_name, sql in queries.items():

        print("\n" + "=" * 65)
        print(query_name)
        print("=" * 65)

        result = pd.read_sql(sql, connection)

        print(result.head(10).to_string(index=False))

    sql_join = queries[
        "Query 7 - Books with their category names (JOIN)"
    ]

    sql_result = pd.read_sql(sql_join, connection)


    books_df = pd.read_sql(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        connection
    )


    pandas_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_result = pandas_result[
        ["title", "price_gbp", "category_name"]
    ]

    sql_result = sql_result[
        ["title", "price_gbp", "category_name"]
    ]


    sql_sorted = sql_result.sort_values(
        by=["title", "category_name"]
    ).reset_index(drop=True)

    pandas_sorted = pandas_result.sort_values(
        by=["title", "category_name"]
    ).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("SQL JOIN VS PANDAS MERGE COMPARISON")
    print("=" * 65)

    print("SQL JOIN rows:", len(sql_result))
    print("Pandas MERGE rows:", len(pandas_result))

    print(
        "Are both outputs equivalent?",
        sql_sorted.equals(pandas_sorted)
    )

    print("\nJOIN OUTPUT PREVIEW")
    print(sql_result.head(10).to_string(index=False))

    connection.close()


if __name__ == "__main__":
    run_queries()
