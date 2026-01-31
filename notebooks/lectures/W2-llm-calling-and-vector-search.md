### Część 1: Klasyfikacja FAQ za pomocą LLM

  - Import bibliotek: pandas, openai, chromadb, sklearn, plotly, tqdm, langchain
  - Załadowanie zmiennych środowiskowych z .env (OPENAI_API_KEY)
  - Wczytanie danych FAQ z travel_company_faq.json (100 pytań z kategorią)
  - Stworzenie system prompt do klasyfikacji pytań do 5 kategorii: air-travel, hotels-and-booking, food, insurance, extra-activities   
  - Funkcja classify_question() - wysyła pytanie do GPT-4o-mini, parsuje JSON response
  - Klasyfikacja wszystkich pytań z użyciem progress_apply
  - Obliczenie accuracy za pomocą sklearn
  - Wizualizacja confusion matrix za pomocą Plotly

### Część 2: Few-Shot Learning

  - Funkcja add_few_shot_examples() - dodaje 2 przykłady per kategoria do system prompt
  - Ponowna klasyfikacja z rozszerzonym promptem
  - Porównanie accuracy - few-shot daje 97% vs podstawowy prompt

### Część 3: Vector Search (ChromaDB)

  - Inicjalizacja ChromaDB - PersistentClient z lokalnym folderem chroma_db
  - Konfiguracja embeddingów - OpenAI text-embedding-ada-002
  - Funkcja ingest_faq_data() - wektoryzuje pytania+odpowiedzi, zapisuje z metadanymi (question, answer, category)
  - Indeksowanie 100 rekordów do kolekcji travel-company-faq
  - Podgląd embeddingów - wektor 1536 floatów per dokument

### Część 4: Wyszukiwanie semantyczne

  - Funkcja retrieve_similar_qas() - wyszukuje n najbardziej podobnych FAQ do zapytania
  - Test wyszukiwania - "What is the air travel lost baggage policy?" → zwraca 3 podobne pytania z distances

### Część 5: Basic RAG

  - Funkcja format_context() - formatuje dokumenty jako HTML-like context
  - Funkcja basic_rag_pipeline():
    a. Retrieve top-n podobnych dokumentów z ChromaDB
    b. Zbuduj system prompt z kontekstem
    c. Wyślij zapytanie do LLM
    d. Zwróć odpowiedź
  - Test RAG - LLM generuje odpowiedź na podstawie pobranych dokumentów

### Część 6: Reranking

  - Import CrossEncoder z sentence-transformers
  - Model reranker: mixedbread-ai/mxbai-rerank-xsmall-v1
  - Funkcja rerank_and_limit_context() - reranking dokumentów, filtrowanie po min_score_threshold
  - Funkcja rag_pipeline_with_reranking() - RAG z dodatkowym krokiem reranking przed wysłaniem do LLM

  ---
  Główne koncepcje w notebooku:
  1. Zero-shot vs Few-shot classification
  2. Vector embeddings i similarity search
  3. RAG (Retrieval Augmented Generation)
  4. Reranking dla poprawy jakości kontekstu