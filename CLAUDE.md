# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trip recommendation system based on NER (Named Entity Recognition) and RAG (Retrieval Augmented Generation). Extracts geographic entities from customer hotel reviews and recommends matching trips using vector similarity search.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md
```

## Key Dependencies

- **spaCy**: NER model for extracting geographic entities
- **ChromaDB**: Vector store for trip embeddings
- **OpenAI API**: `text-embedding-3-small` for embeddings
- **python-dotenv**: Environment variable management

## Required Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key
```

## Project Structure

```
scripts/
└── ner-trip-recommender.py    # Main recommendation pipeline

models/
└── ner_geo/                   # Trained spaCy NER model (geo entities)

notebooks/
├── Get-NER.ipynb              # NER extraction from reviews
└── RAG-Trip-Recommendations.ipynb  # Vector search experiments

data/
├── trips_data.json            # 139 trips (Country, City, Cost, Details)
├── customer_surveys_hotels_1k.json    # 1000 hotel reviews (input)
├── customer_surveys_hotels_1k_ner.json # Reviews with extracted entities
└── trip_recommendations.json  # Generated recommendations
```

## Main Pipeline

```
Customer Review (text)
    ↓ spaCy NER (models/ner_geo)
GEO Entities (geo, gpe, nat)
    ↓ OpenAI Embeddings
Query Vector
    ↓ ChromaDB Similarity Search
Top 3 Trip Recommendations
```

## Running the Recommender

```bash
cd scripts
python ner-trip-recommender.py
```

## Key Functions (ner-trip-recommender.py)

| Function | Description |
|----------|-------------|
| `load_spacy_model(path)` | Loads trained NER model from disk |
| `extract_entities_from_review(text)` | Extracts NER entities from review |
| `initialize_trip_vector_store()` | Creates ChromaDB collection with trips |
| `search_trips_by_entities(entities, n)` | Vector search for matching trips |
| `recommend_trips_from_review(review)` | **End-to-end pipeline**: review → recommendations |

## NER Entity Types

| Label | Description | Example |
|-------|-------------|---------|
| `geo` | Geographic location | "las ramblas", "red sea" |
| `gpe` | Geopolitical entity | "spain", "egypt" |
| `nat` | Natural landmark | "coral reef" |

## Data Formats

### trips_data.json
```json
{
  "Country": "Spain",
  "City": "Barcelona",
  "Start date": "2025-04-02",
  "Count of days": 4,
  "Cost in EUR": 950,
  "Extra activities": ["City tour", "Food tasting"],
  "Trip details": "Discover Barcelona's famous architecture..."
}
```

### Recommendation Output
```json
{
  "rank": 1,
  "country": "Spain",
  "city": "Barcelona",
  "cost": 950,
  "days": 4,
  "distance": 0.35,
  "details": "..."
}
```
