# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trip recommendation system based on NER (Named Entity Recognition), RAG (Retrieval Augmented Generation), and LLM-powered sentiment analysis. Extracts geographic entities from customer hotel reviews, classifies sentiment, and either recommends matching trips (positive) or generates a personalized apology with a 25% discount offer (negative). The full pipeline is composed as a single LangChain chain using `RunnablePassthrough` and `RunnableBranch`.

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
- **OpenAI API**: `text-embedding-3-small` for embeddings, `gpt-4o` for sentiment & response generation
- **LangChain**: Chains, prompts, output parsers (`RunnablePassthrough`, `RunnableBranch`, `RunnableLambda`)
- **Streamlit**: Web UI for interactive review analysis
- **scikit-learn**: Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- **Plotly**: Interactive charts (confusion matrix heatmap, metrics bar charts)
- **python-dotenv**: Environment variable management

## Required Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key
```

## Project Structure

```
scripts/
├── ner-trip-recommender.py    # Main recommendation + sentiment pipeline
└── app.py                     # Streamlit web UI

models/
└── ner_geo/                   # Trained spaCy NER model (geo entities)

notebooks/
├── Get-NER.ipynb              # NER extraction from reviews
├── RAG-Trip-Recommendations.ipynb  # Vector search experiments
├── W3-llm-flows-and-monitoring.ipynb  # Sentiment + LLM routing chains
└── Sentiment-Classification.ipynb    # Batch sentiment classification (1k reviews) + evaluation metrics

data/
├── trips_data.json            # 139 trips (Country, City, Cost, Details)
├── customer_surveys_hotels_1k.json    # 1000 hotel reviews (id, review, score, sentiment)
├── customer_surveys_hotels_1k_ner.json # Reviews with extracted entities
├── trip_recommendations.json  # Generated recommendations
└── sentiment_classification_results.json  # Batch classification output (predicted vs survey)
```

## Full Review Chain Pipeline

```
Customer Review (text)
    ↓ sentiment_chain (GPT-4o)
Sentiment {positive_sentiment, reasoning}
    ↓ RunnableBranch
    ├─ POSITIVE:
    │   ↓ spaCy NER (models/ner_geo)
    │   GEO Entities (geo, gpe, nat)
    │   ↓ OpenAI Embeddings + ChromaDB
    │   Top 3 Trip Recommendations
    │   ↓ positive_response_chain (GPT-4o)
    │   Personalized message with trip suggestion
    │
    └─ NEGATIVE:
        ↓ spaCy NER (models/ner_geo)
        GEO Entities (geo, gpe, nat)
        ↓ negative_response_chain (GPT-4o)
        Apology + 25% discount offer
```

## Running

```bash
# Console test
python scripts/ner-trip-recommender.py

# Streamlit web app
streamlit run scripts/app.py
```

## Key Functions (ner-trip-recommender.py)

| Function | Description |
|----------|-------------|
| `analyze_sentiment(review)` | Classifies review as positive/negative via LLM |
| `handle_positive_review(review)` | Sentiment + NER extraction → trip recommendations → personalized message |
| `handle_negative_review(review)` | Sentiment check → apology with 25% discount |
| `extract_entities_from_review(text)` | Extracts NER entities from review via spaCy |
| `load_spacy_model(path)` | Loads trained NER model from disk |
| `initialize_trip_vector_store()` | Creates ChromaDB collection with trips |
| `search_trips_by_entities(entities, n)` | Vector search for matching trips |
| `recommend_trips_from_review(review)` | NER → vector search → top N recommendations |
| `full_review_chain` | **Combined LangChain chain**: review → sentiment → branch → response |

## LangChain Chains

| Chain | Purpose |
|-------|---------|
| `sentiment_chain` | `PromptTemplate` → `ChatOpenAI` → `JsonOutputParser` → `{positive_sentiment, reasoning}` |
| `negative_response_chain` | Generates apology addressing specific issues + 25% discount |
| `positive_response_chain` | Generates thank-you message referencing recommended trips |
| `full_review_chain` | `RunnablePassthrough.assign` (sentiment) → `RunnableBranch` (positive/negative routing) |

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

### full_review_chain Output (positive)
```json
{
  "positive_sentiment": true,
  "reasoning": "The review uses positive language such as 'amazing', 'incredible', 'friendly'...",
  "entities": [{"label": "gpe", "text": "egypt"}, {"label": "geo", "text": "red sea"}],
  "recommendations": [
    {
      "rank": 1,
      "country": "Egypt",
      "city": "Hurghada",
      "cost": 1200,
      "days": 7,
      "start_date": "2025-06-01",
      "extra_activities": ["Snorkeling", "Desert safari"],
      "distance": 0.35,
      "details": "..."
    }
  ],
  "response_message": "Thank you for your wonderful feedback..."
}
```

### sentiment_classification_results.json
```json
{
  "id": "7a823fbc-e97b-4e8b-8fb0-f60dfb79a8ef",
  "review": "hotel america nice hotel good location...",
  "customer_satisfaction_score": 4,
  "survey_sentiment": "positive",
  "predicted_sentiment": "positive"
}
```

## Sentiment Evaluation (Sentiment-Classification.ipynb)

The notebook loads `customer_surveys_hotels_1k.json`, classifies each review with `analyze_sentiment` (GPT-4o, binary positive/negative), and evaluates against the `survey_sentiment` labels (3-class: positive/negative/neutral).

**Evaluation:** Binary only — 200 neutral reviews excluded for a fair pos/neg comparison on 800 samples.

**Metrics computed:** Accuracy, Precision, Recall, F1-score, full classification report (per class + weighted average)

**Visualization (Plotly):**
- 2×2 confusion matrix heatmap (binary, neutral excluded, Blues colorscale)

### full_review_chain Output (negative)
```json
{
  "positive_sentiment": false,
  "reasoning": "The review expresses dissatisfaction...",
  "entities": [{"label": "gpe", "text": "italy"}],
  "recommendations": [],
  "response_message": "We sincerely apologize... 25% discount on your next trip..."
}
```
