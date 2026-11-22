
# Zepto Data & AI Platform

An end-to-end Data and AI project containing a web-scraping pipeline, exploratory data analysis and machine-learning models, and an AI-powered customer support assistant.

## Project Modules

### 1. Data Pipeline
- Scrapes book information from Books to Scrape.
- Extracts titles, prices, ratings, availability and categories.
- Cleans and transforms the collected data.
- Stores normalized data in an SQLite database.
- Executes SQL queries and compares SQL JOIN results with pandas merge.

**Folder:** `data_pipeline/`

### 2. Analytics and Machine Learning
- Performs exploratory data analysis on the Titanic dataset.
- Analyzes missing values, distributions, outliers and correlations.
- Builds Logistic Regression, Decision Tree and Random Forest classifiers.
- Evaluates classification and regression models.
- Performs hyperparameter tuning and saves a fitted model.

**Folder:** `analytics/`

### 3. AI Support Assistant
- Uses eight Zepto policy documents as its knowledge base.
- Creates embeddings using Sentence Transformers.
- Stores and retrieves relevant information using ChromaDB.
- Uses LangGraph for intent classification and response routing.
- Provides a FastAPI endpoint for customer questions.
- Supports deterministic offline mock mode and optional real LLM mode.

**Folder:** `support_assistant/`

## Repository Structure

```text
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── queries.py
│   ├── main.py
│   └── README.md
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   ├── ingest.py
│   ├── schemas.py
│   ├── graph.py
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
├── requirements.txt
└── README.md
```

## Installation

Python 3.11 or later is recommended.

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running the Modules

### Data Pipeline

```bash
cd data_pipeline
python main.py
```

### Analytics

Run the scripts from the analytics directory:

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

### AI Support Assistant

Run these commands from the support_assistant directory:

```bash
cd support_assistant
python ingest.py
uvicorn main:app --reload
```

Open the API documentation at:

`http://127.0.0.1:8000/docs`

## Technologies Used

- Python
- Requests and BeautifulSoup
- SQLite and SQL
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn and Imbalanced-learn
- Sentence Transformers
- ChromaDB
- LangGraph
- FastAPI
- Docker

## Project Objective

The objective is to demonstrate an end-to-end workflow covering data collection, database management, data analysis, machine learning, document retrieval and AI-powered customer support.

## Author

Developed as a Data and AI Platform project.
