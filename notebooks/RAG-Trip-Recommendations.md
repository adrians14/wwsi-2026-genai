### 0. Setup

- Import chromadb, openai
- Załadowanie .env z API key

### 1. Load trips data

- Wczytanie 139 wycieczek z trips_data.json

### 2. Prepare texts for embedding

- Funkcja trip_to_text() - łączy Country, City, Trip details, Activities

### 3. Create Vector Store

- ChromaDB in-memory
- OpenAI text-embedding-3-small jako embedding function

### 4. Index trips

- Dodanie wszystkich wycieczek do ChromaDB z metadatami (cost, days, etc.)

### 5. Search function

- search_trips_by_entities(geo_entities) - główna funkcja wyszukiwania
- print_recommendations() - ładne wyświetlanie wyników

### 6. Test queries

- Test z encjami: Spain/Barcelona, Rome/Italy, Beach/Snorkeling

### 7. Integration with NER

- Wczytanie wyników NER z customer_surveys_hotels_1k_ner.json
- recommend_trips_for_review() - rekomendacje na podstawie encji z recenzji

### 8. Batch processing

- Przetworzenie wszystkich recenzji
- Zapis do trip_recommendations.json