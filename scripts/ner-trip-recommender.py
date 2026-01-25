import os
import json
import spacy
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# NER MODEL SETUP (from Get-NER.ipynb)
# =============================================================================

def load_spacy_model(model_path, base_model="en_core_web_md"):
    """Load a trained spaCy NER model from disk."""
    nlp = spacy.load(base_model)
    file = open(f'{model_path}/bytes_data.bin', "rb")
    bytes_data = file.read()
    config = nlp.config
    lang_cls = spacy.util.get_lang_class("en")
    nlp = lang_cls.from_config(config)
    nlp = nlp.from_disk(f'{model_path}')
    return nlp


# Load NER model
print("Loading NER model...")
ner_model = load_spacy_model("../models/ner_geo")
print("NER model loaded.")


# =============================================================================
# EXTRACT NER FROM SINGLE REVIEW (from Get-NER.ipynb)
# =============================================================================

def extract_entities_from_review(review_text: str) -> dict:
    """
    Extract NER entities from a single review.

    Args:
        review_text: The customer review text

    Returns:
        dict with 'text' and 'entities' fields
    """
    doc = ner_model(review_text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "label": ent.label_,
            "text": ent.text
        })

    return {
        "text": review_text,
        "entities": entities
    }


# =============================================================================
# CHROMADB VECTOR STORE SETUP (from RAG-Trip-Recommendations.ipynb)
# =============================================================================

def trip_to_text(trip: dict) -> str:
    """Convert trip dict to text for embedding."""
    activities = ", ".join(trip["Extra activities"])
    return f"{trip['Country']}, {trip['City']}. {trip['Trip details']} Activities: {activities}"


def initialize_trip_vector_store():
    """Initialize ChromaDB with trips data."""
    print("Initializing ChromaDB vector store...")

    # Load trips data
    with open('../data/trips_data.json', 'r', encoding='utf-8') as f:
        trips = json.load(f)

    # Initialize ChromaDB (in-memory)
    chroma_client = chromadb.Client()

    # Create embedding function using OpenAI
    embedding_fn = OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small"
    )

    # Create collection
    collection = chroma_client.get_or_create_collection(
        name="trips",
        embedding_function=embedding_fn
    )

    # Prepare texts and metadata
    trip_texts = [trip_to_text(trip) for trip in trips]
    trip_ids = [f"trip_{i}" for i in range(len(trips))]
    trip_metadatas = [
        {
            "country": trip["Country"],
            "city": trip["City"],
            "cost": trip["Cost in EUR"],
            "days": trip["Count of days"],
            "start_date": trip["Start date"]
        }
        for trip in trips
    ]

    # Add to collection
    collection.add(
        documents=trip_texts,
        ids=trip_ids,
        metadatas=trip_metadatas
    )

    print(f"Indexed {collection.count()} trips.")
    return collection, trips


# Initialize vector store
collection, trips = initialize_trip_vector_store()


# =============================================================================
# SEARCH FUNCTIONS (from RAG-Trip-Recommendations.ipynb)
# =============================================================================

def search_trips_by_entities(geo_entities: list, n_results: int = 3) -> list:
    """
    Search for trips based on GEO entities from customer review.

    Args:
        geo_entities: list of GEO entity strings, e.g. ["spain", "las ramblas"]
        n_results: number of trips to return

    Returns:
        list of matching trips with scores
    """
    # Convert entities to natural query
    query = f"Trip visiting: {', '.join(geo_entities)}"

    # Search in ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Format results
    recommendations = []
    for i in range(len(results['ids'][0])):
        trip_id = results['ids'][0][i]
        trip_idx = int(trip_id.split('_')[1])

        recommendations.append({
            "rank": i + 1,
            "country": results['metadatas'][0][i]['country'],
            "city": results['metadatas'][0][i]['city'],
            "cost": results['metadatas'][0][i]['cost'],
            "days": results['metadatas'][0][i]['days'],
            "distance": results['distances'][0][i],
            "details": trips[trip_idx]['Trip details']
        })

    return recommendations


def print_recommendations(recommendations: list, entities: list):
    """Pretty print recommendations."""
    print(f"\n{'='*60}")
    print(f"GEO Entities: {entities}")
    print(f"{'='*60}")

    for rec in recommendations:
        print(f"\n#{rec['rank']}: {rec['city']}, {rec['country']}")
        print(f"   Cost: {rec['cost']} EUR | Days: {rec['days']}")
        print(f"   Score: {1 - rec['distance']:.3f}")
        print(f"   {rec['details'][:200]}...")


# =============================================================================
# RECOMMEND TRIPS FROM REVIEW TEXT
# =============================================================================

def recommend_trips_from_review(review: str, n_results: int = 3, verbose: bool = True) -> list:
    """
    End-to-end function: takes a review text and returns trip recommendations.

    Pipeline:
    1. Extract NER entities from review
    2. Filter GEO-related entities (geo, gpe, nat)
    3. Search for matching trips using vector similarity
    4. Return top N recommendations

    Args:
        review: Customer review text
        n_results: Number of trips to recommend (default: 3)
        verbose: Whether to print intermediate results

    Returns:
        list of trip recommendations with details
    """
    if verbose:
        print(f"\n{'='*60}")
        print("INPUT REVIEW:")
        print(f"{'='*60}")
        print(f"{review[:300]}..." if len(review) > 300 else review)

    # Step 1: Extract NER entities
    ner_result = extract_entities_from_review(review)

    if verbose:
        print(f"\n> Extracted entities: {ner_result['entities']}")

    # Step 2: Filter GEO-related entities
    geo_entities = [
        ent['text']
        for ent in ner_result['entities']
        if ent['label'] in ['geo', 'gpe', 'nat']
    ]

    # Remove duplicates while preserving order
    geo_entities = list(dict.fromkeys(geo_entities))

    if not geo_entities:
        if verbose:
            print("\n> No GEO entities found in this review.")
        return []

    if verbose:
        print(f"> GEO entities for search: {geo_entities}")

    # Step 3: Search for matching trips
    recommendations = search_trips_by_entities(geo_entities, n_results)

    # Step 4: Print and return results
    if verbose:
        print_recommendations(recommendations, geo_entities)

    return recommendations


# =============================================================================
# MAIN - TEST THE PIPELINE
# =============================================================================

if __name__ == "__main__":
    # Test review 1: Spanish hotel
    review_1 = """hotel america nice hotel good location stayed 3 nights hotel america late december,
    rooms modern nice, really liked location hotel, located 3 blocks main area, excellent location
    base stay explore interesting parts city, able walk las ramblas neighborhoods gothic district,
    walked sacred familia cathedral no 15 minutes morning, breakfast adequate run things wait long
    breakfast, negatives street noise pretty loud room, set typical spain hard fault hotel america"""

    print("\n" + "="*80)
    print("TEST 1: Spanish Hotel Review")
    print("="*80)
    recommendations_1 = recommend_trips_from_review(review_1)

    # Test review 2: Beach resort
    review_2 = """amazing beach resort red sea, snorkeling was incredible, saw beautiful coral reef
    and colorful fish, staff very friendly, food was great, egyptian cuisine delicious,
    visited pyramids on day trip to cairo, unforgettable experience in egypt"""

    print("\n" + "="*80)
    print("TEST 2: Egypt Beach Resort Review")
    print("="*80)
    recommendations_2 = recommend_trips_from_review(review_2)

    # Test review 3: Italian trip
    review_3 = """wonderful stay in rome, visited colosseum and vatican, ate amazing pasta and gelato,
    italian food is the best, walked through beautiful piazzas, romantic city"""

    print("\n" + "="*80)
    print("TEST 3: Italian Trip Review")
    print("="*80)
    recommendations_3 = recommend_trips_from_review(review_3)
