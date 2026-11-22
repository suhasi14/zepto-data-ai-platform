from scraper import main as run_scraper
from database import main as run_database
from queries import run_queries


def main():

    print("\n" + "=" * 60)
    print("STEP 1: WEB SCRAPING AND DATA CLEANING")
    print("=" * 60)

    run_scraper()

    print("\n" + "=" * 60)
    print("STEP 2: DATABASE CREATION AND DATA INSERTION")
    print("=" * 60)

    run_database()

    print("\n" + "=" * 60)
    print("STEP 3: SQL QUERIES AND PANDAS COMPARISON")
    print("=" * 60)

    run_queries()

    print("\n" + "=" * 60)
    print("COMPLETE DATA PIPELINE EXECUTION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
