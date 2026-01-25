# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational repository for GenAI coursework (WWSI 2026) focused on NLP tasks: sentiment analysis, Named Entity Recognition (NER), LLM workflow patterns using LangChain, and RAG-based trip recommendations.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install chromadb  # For RAG notebook

# Download spaCy models (required for NER notebooks)
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

## Key Dependencies

- **LangChain** (`langchain-core`, `langchain-openai`): LLM orchestration and prompt chaining
- **spaCy**: NER model training and inference
- **OpenAI API**: GPT-4o for sentiment, `text-embedding-3-small` for embeddings
- **ChromaDB**: Vector store for RAG trip recommendations
- **LangSmith**: LLM tracing and monitoring (configured via `.env`)

## Required Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name
```

## Project Structure

```
notebooks/
├── W2-NER-Intro.ipynb           # NER training with spaCy (GMB dataset)
├── W3-llm-flows-and-monitoring.ipynb  # LangChain sentiment + routing
├── Get-NER.ipynb                # Extract NER from customer reviews
└── RAG-Trip-Recommendations.ipynb     # Vector search trip recommendations

data/
├── GMB_data_*.pickle            # NER training data (spaCy format)
├── customer_surveys_hotels_1k.json    # 1000 hotel reviews (input)
├── customer_surveys_hotels_1k_ner.json # Reviews with extracted entities
├── trips_data.json              # 139 trips (Country, City, Cost, Details)
└── trip_recommendations.json    # Generated recommendations
```

## Workflow Pipeline

```
1. customer_surveys_hotels_1k.json
   ↓ (Get-NER.ipynb + spaCy NER model)
2. customer_surveys_hotels_1k_ner.json (reviews + GEO entities)
   ↓ (RAG-Trip-Recommendations.ipynb + ChromaDB)
3. trip_recommendations.json (3 best trips per review)
```

## Architecture Patterns

### LangChain Flow Pattern (W3 notebook)
1. **Sentiment Chain**: Review → JsonOutputParser → `{positive_sentiment: bool, reasoning: str}`
2. **RunnableBranch**: Routes to positive_chain (voucher) or negative_chain (25% discount)
3. Chains use `RunnablePassthrough.assign()` to preserve context

### NER Training Pattern (W2 notebook)
1. Data format: `(text, {"entities": [(start, end, label), ...]})`
2. Entity types: `per`, `org`, `geo`, `gpe`, `tim`, `art`, `eve`, `nat`
3. Training uses `nlp.update()` with disabled non-NER pipes
4. Models saved via `nlp.to_disk()` with bytes serialization

### RAG Trip Recommendations (RAG notebook)
1. **Indexing**: trips_data.json → OpenAI embeddings → ChromaDB vector store
2. **Query**: GEO entities → "Trip visiting: {entities}" → embedding → similarity search
3. **Result**: Top 3 trips by cosine similarity with metadata (cost, days, city)

## Key Functions

### Get-NER.ipynb
- `text_to_entities(text, nlp)` → `(text, {"entities": [(start, end, label, text), ...]})`
- `save_entities_to_json(ent_result, output_path)` → saves to JSON

### RAG-Trip-Recommendations.ipynb
- `trip_to_text(trip)` → combines trip fields for embedding
- `search_trips_by_entities(geo_entities, n_results=3)` → returns top matching trips
- `recommend_trips_for_review(ner_result)` → end-to-end recommendation

## Running Notebooks

Run from `notebooks/` directory. Execution order:
1. `W2-NER-Intro.ipynb` - Train/load NER model
2. `Get-NER.ipynb` - Extract entities from reviews
3. `RAG-Trip-Recommendations.ipynb` - Generate trip recommendations
