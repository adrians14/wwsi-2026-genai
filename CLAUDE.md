# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational repository for GenAI coursework (WWSI 2026) focused on NLP tasks: sentiment analysis, Named Entity Recognition (NER), and LLM workflow patterns using LangChain.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for NER notebooks)
python -m spacy download en_core_web_md
```

## Key Dependencies

- **LangChain** (`langchain-core`, `langchain-openai`): LLM orchestration and prompt chaining
- **spaCy**: NER model training and inference
- **OpenAI API**: GPT-4o for sentiment analysis and response generation
- **LangSmith**: LLM tracing and monitoring (configured via `.env`)

## Required Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name
```

## Project Structure

- `notebooks/` - Jupyter notebooks for coursework
  - `W2-NER-Intro.ipynb` - NER training with spaCy using GMB dataset
  - `W3-llm-flows-and-monitoring.ipynb` - LangChain sentiment analysis with routing
- `data/` - Training data and pickle files
  - `GMB_data_*.pickle` - NER training data in spaCy format
  - `customer_surveys*.json` - Customer review data for sentiment analysis
  - `trips_data.json` - Trip recommendations data

## Architecture Patterns

### LangChain Flow Pattern (W3 notebook)
1. **Sentiment Chain**: Review → JsonOutputParser → `{positive_sentiment: bool, reasoning: str}`
2. **RunnableBranch**: Routes to positive_chain (voucher offer) or negative_chain (25% discount + apology)
3. Chains use `RunnablePassthrough.assign()` to preserve context through pipeline

### NER Training Pattern (W2 notebook)
1. Data format: `(text, {"entities": [(start, end, label), ...]})`
2. Entity types: `per` (person), `org` (organization), `geo`, `gpe`, `tim`, `art`, `eve`, `nat`
3. Training uses `nlp.update()` with disabled non-NER pipes
4. Models saved via `nlp.to_disk()` with bytes serialization

## Running Notebooks

Use Jupyter or VS Code notebook support. Notebooks expect to be run from the `notebooks/` directory (relative paths like `../data/` are used).
