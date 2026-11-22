
# Module 1: Data Pipeline

## 1. Project Overview

This module collects book information from the Books to Scrape website, cleans the collected data, converts prices from GBP to INR, stores the information in a normalized SQLite database, and performs SQL-based analysis.

## 2. Website Used

Website: https://books.toscrape.com/

Scraping scope: First five catalogue pages.

Tools used:
- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite3

## 3. Data Fields Collected

The scraper collects the following fields:

| Field | Description |
|---|---|
| title | Name of the book |
| price | Original displayed price |
| star_rating | Written star rating |
| availability | Availability text |
| category | Book category |

## 4. Data Cleaning

The following cleaning operations are performed:

1. Price strings are converted into numeric GBP values.
2. Written star ratings are converted into integers from 1 to 5.
3. Availability is converted into a Boolean in_stock field.
4. GBP prices are converted into INR using the fixed assignment rate of 1 GBP = INR 105.50.
5. Rows with missing required values are removed.

## 5. Database Design

The project uses SQLite with two related tables.

### Categories Table

- category_id: Primary key
- category_name: Unique category name

### Books Table

- book_id: Primary key
- title: Book title
- price_gbp: Price in GBP
- price_inr: Price in INR
- rating: Numerical star rating
- availability: Availability description
- in_stock: Boolean stock status
- category_id: Foreign key referencing Categories

The relationship between the tables is one-to-many: one category can contain multiple books.

## 6. SQL Queries

The project demonstrates:

1. SELECT – Retrieve book information.
2. WHERE – Filter books by price.
3. ORDER BY – Sort books by price.
4. LIMIT – Retrieve the five most expensive books.
5. DISTINCT – Retrieve unique categories.
6. BETWEEN – Filter books within a price range.
7. JOIN – Combine books and categories.

## 7. Pandas JOIN Comparison

The SQL JOIN result is reproduced using pandas.merge().

The project compares both outputs to verify that they contain equivalent records.

## 8. Project Files

- scraper.py: Web scraping and data cleaning.
- database.py: SQLite database creation and insertion.
- queries.py: SQL queries and pandas comparison.
- main.py: Complete pipeline execution.

## 9. Execution Order

Run the complete pipeline from the data_pipeline directory:

    python main.py

Install the required dependencies before execution.

## 10. Expected Outputs

- data/raw_books.csv
- data/cleaned_books.csv
- books_database.db

The outputs are generated when the scripts are executed locally.
