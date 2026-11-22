import os
import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DataPipelineProject/1.0)"
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

def get_soup(url):
    """Download a webpage and return its BeautifulSoup object."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def clean_price(price_text):
    """Convert a price such as £51.77 into a float."""

    cleaned = re.sub(r"[^0-9.]", "", price_text)

    return float(cleaned)


def get_category(book_url):
    """Extract the actual book category from its detail page."""

    try:
        soup = get_soup(book_url)

        breadcrumbs = soup.select(
            "ul.breadcrumb li a"
        )

        # The second breadcrumb link represents the category.
        if len(breadcrumbs) >= 3:
            return breadcrumbs[2].get_text(strip=True)

        return "Unknown"

    except (requests.RequestException, AttributeError):
        return "Unknown"

def scrape_books():

    books = []

    for page_number in range(1, 6):

        if page_number == 1:
            page_url = BASE_URL
        else:
            page_url = urljoin(
                BASE_URL,
                f"catalogue/page-{page_number}.html"
            )

        print(f"Scraping page {page_number}: {page_url}")

        try:
            soup = get_soup(page_url)

        except requests.RequestException as error:
            print(f"Could not scrape page {page_number}: {error}")
            continue

        book_cards = soup.select("article.product_pod")

        for card in book_cards:

            try:
                # Book title
                title_element = card.select_one("h3 a")
                title = title_element.get("title", "").strip()

                # Book detail URL
                relative_url = title_element.get("href")
                book_url = urljoin(page_url, relative_url)

                # Price in GBP (original text)
                price_text = card.select_one(
                    ".price_color"
                ).get_text(strip=True)

                # Star rating as text
                rating_element = card.select_one(
                    "p.star-rating"
                )

                star_rating = next(
                    (
                        class_name
                        for class_name in rating_element.get("class", [])
                        if class_name in RATING_MAP
                    ),
                    None
                )

                # Availability as listed text
                availability = card.select_one(
                    ".availability"
                ).get_text(" ", strip=True)

                # Actual book category
                category = get_category(book_url)

                books.append({
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category
                })

                time.sleep(0.1)

            except (AttributeError, TypeError, ValueError) as error:
                print(f"Skipping a book because of a parsing error: {error}")
                continue

    return books

def clean_books(books):

    df = pd.DataFrame(books)

    if df.empty:
        raise ValueError("No books were scraped. Check your internet connection.")

    df["price_gbp"] = pd.to_numeric(
        df["price"].apply(clean_price),
        errors="coerce"
    )

    df["rating"] = df["star_rating"].map(RATING_MAP)

    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )
  
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)
  
    initial_count = len(df)

    df = df.dropna(
        subset=[
            "title",
            "price_gbp",
            "rating",
            "category",
            "price_inr"
        ]
    ).copy()

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    removed_count = initial_count - len(df)

    print(f"\nRows removed due to invalid data: {removed_count}")

    return df


def main():

    print("=" * 55)
    print("ZEpto DATA & AI PLATFORM - DATA PIPELINE")
    print("=" * 55)

    scraped_books = scrape_books()

    print(f"\nTotal books scraped: {len(scraped_books)}")

    if not scraped_books:
        print("No books collected. Please try again.")
        return

    # Save original scraped data.
    raw_df = pd.DataFrame(scraped_books)

    raw_path = os.path.join(
        OUTPUT_DIR,
        "raw_books.csv"
    )

    raw_df.to_csv(raw_path, index=False)

    # Clean the scraped data.
    cleaned_df = clean_books(scraped_books)

    cleaned_path = os.path.join(
        OUTPUT_DIR,
        "cleaned_books.csv"
    )

    cleaned_df.to_csv(cleaned_path, index=False)

    print("\nCLEANED DATA PREVIEW")
    print(cleaned_df.head(10).to_string(index=False))

    print("\nDATASET SUMMARY")
    print(f"Total cleaned books: {len(cleaned_df)}")
    print(f"Number of categories: {cleaned_df['category'].nunique()}")
    print(f"GBP to INR conversion rate: {GBP_TO_INR}")

    print("\nFiles saved:")
    print(raw_path)
    print(cleaned_path)

    if len(cleaned_df) >= 60 and cleaned_df["category"].nunique() >= 3:
        print("\nSUCCESS: Minimum assignment scraping requirements satisfied.")
    else:
        print("\nWARNING: Check whether the dataset meets the minimum requirements.")

    print("\nDATA PIPELINE SCRAPING COMPLETED.")


if __name__ == "__main__":
    main()
