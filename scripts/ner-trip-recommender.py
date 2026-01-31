import os
import json
import spacy
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))


# =============================================================================
# SENTIMENT ANALYSIS (from W3-llm-flows-and-monitoring.ipynb)
# =============================================================================

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

sentiment_prompt = PromptTemplate(
    template="""You are a sentiment analysis expert.
Review the following customer review and determine if it's positive or negative.

Review: ```{review}```

Return answer as a valid json object with the following format:
{{"positive_sentiment": boolean, "reasoning": string}}
""",
    input_variables=["review"]
)

sentiment_chain = sentiment_prompt | llm | JsonOutputParser()


def analyze_sentiment(review: str) -> dict:
    """
    Analyze sentiment of a customer review using LLM.

    Args:
        review: Customer review text

    Returns:
        dict with keys:
            - positive_sentiment (bool): True if review is positive
            - reasoning (str): Explanation of the sentiment classification
    """
    return sentiment_chain.invoke({"review": review})


# =============================================================================
# NEGATIVE REVIEW RESPONSE (from W3-llm-flows-and-monitoring.ipynb)
# =============================================================================

negative_response_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer has left a negative review about one of our trips.

Customer Review: {review}

Based on the review, identify what they specifically disliked and create a personalized response that:
1. Apologizes for their negative experience
2. Addresses the specific issue they mentioned
3. Explains how you'll mitigate this issue in the future
4. Offers a 25% discount on their next trip
5. Thanks them for their feedback

Return your response as a valid JSON object with the following format:
{{"message": str}}
""",
    input_variables=["review"]
)

negative_response_chain = negative_response_prompt | llm | JsonOutputParser()


# =============================================================================
# POSITIVE REVIEW RESPONSE (from W3-llm-flows-and-monitoring.ipynb)
# =============================================================================

positive_response_prompt = PromptTemplate(
    template="""You are a customer service representative for a travel company.
A customer has left a positive review about one of our trips.

Customer Review: {review}

Based on the review and the recommended trips below, create a personalized response that:
1. Thanks them for their positive feedback
2. Highlights what they enjoyed
3. Recommends the top trip below that best matches their interests
4. Encourages them to book their next adventure

Recommended trips:
{recommendations_text}

Return your response as a valid JSON object with the following format:
{{"message": str}}
""",
    input_variables=["review", "recommendations_text"]
)

positive_response_chain = positive_response_prompt | llm | JsonOutputParser()


def handle_negative_review(review: str) -> dict:
    """
    Analyze review sentiment and generate a personalized apology for negative reviews.

    If the review is positive, returns None for the response message.

    Args:
        review: Customer review text

    Returns:
        dict with keys:
            - positive_sentiment (bool): True if review is positive
            - reasoning (str): Explanation of the sentiment classification
            - response_message (str | None): Apology message with 25% discount offer,
              or None if review is positive
    """
    sentiment = analyze_sentiment(review)

    response_message = None
    if not sentiment["positive_sentiment"]:
        result = negative_response_chain.invoke({"review": review})
        response_message = result["message"]

    return {
        **sentiment,
        "response_message": response_message,
    }


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
ner_model = load_spacy_model(os.path.join(PROJECT_ROOT, "models", "ner_geo"))
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
    with open(os.path.join(PROJECT_ROOT, 'data', 'trips_data.json'), 'r', encoding='utf-8') as f:
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
            "start_date": results['metadatas'][0][i]['start_date'],
            "extra_activities": trips[trip_idx]['Extra activities'],
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
# POSITIVE REVIEW HANDLER
# =============================================================================

def handle_positive_review(review: str) -> dict:
    """
    Generate trip recommendations and a personalized message for a positive review.

    Args:
        review: Customer review text

    Returns:
        dict with keys:
            - positive_sentiment (bool): always True
            - reasoning (str): explanation of the sentiment classification
            - entities (list): extracted NER entities [{label, text}, ...]
            - recommendations (list): top 3 matching trips
            - response_message (str): personalized message with trip suggestion
    """
    sentiment = analyze_sentiment(review)

    ner_result = extract_entities_from_review(review)
    entities = ner_result["entities"]

    geo_entities = list(dict.fromkeys(
        ent["text"] for ent in entities if ent["label"] in ["geo", "gpe", "nat"]
    ))

    recommendations = search_trips_by_entities(geo_entities) if geo_entities else []

    recs_text = "\n".join([
        f"#{r['rank']}: {r['city']}, {r['country']} - {r['cost']} EUR, {r['days']} days - {r['details'][:150]}"
        for r in recommendations
    ]) or "No specific trip recommendations available."

    response = positive_response_chain.invoke({
        "review": review,
        "recommendations_text": recs_text,
    })

    return {
        "positive_sentiment": True,
        "reasoning": sentiment["reasoning"],
        "entities": entities,
        "recommendations": recommendations,
        "response_message": response["message"],
    }


# =============================================================================
# FULL REVIEW CHAIN (RunnablePassthrough + RunnableBranch)
# =============================================================================
#
# Pipeline:
#   Input: {"review": "..."}
#     ↓ sentiment_chain (classify positive/negative)
#     ↓ RunnableBranch
#     ├─ positive → handle_positive_review (recommendations + personalized message)
#     └─ negative → negative_response_chain (apology + 25% discount)
#

full_review_chain = (
    RunnablePassthrough.assign(
        sentiment_result=sentiment_chain
    )
    | RunnableBranch(
        (
            lambda x: x["sentiment_result"]["positive_sentiment"],
            RunnableLambda(lambda x: handle_positive_review(x["review"]))
        ),
        (
            lambda x: not x["sentiment_result"]["positive_sentiment"],
            RunnableLambda(lambda x: {
                "positive_sentiment": False,
                "reasoning": x["sentiment_result"]["reasoning"],
                "entities": extract_entities_from_review(x["review"])["entities"],
                "recommendations": [],
                "response_message": negative_response_chain.invoke(
                    {"review": x["review"]}
                )["message"],
            })
        ),
        # Default fallback
        RunnableLambda(lambda x: {
            "positive_sentiment": None,
            "reasoning": "Unable to determine sentiment",
            "entities": [],
            "recommendations": [],
            "response_message": None,
        })
    )
)


# =============================================================================
# MAIN - TEST THE PIPELINE
# =============================================================================

if __name__ == "__main__":
    # ---- Test full_review_chain with a positive review ----
    positive_review = """amazing beach resort red sea, snorkeling was incredible, saw beautiful coral reef
    and colorful fish, staff very friendly, food was great, egyptian cuisine delicious,
    visited pyramids on day trip to cairo, unforgettable experience in egypt"""

    print("\n" + "="*80)
    print("FULL CHAIN TEST 1: Positive Review (Egypt)")
    print("="*80)
    print(f"Review: {positive_review.strip()[:200]}...")

    result_positive = full_review_chain.invoke({"review": positive_review})

    print(f"\nPositive? {result_positive['positive_sentiment']}")
    if result_positive.get("entities"):
        print(f"\nEntities:")
        for ent in result_positive["entities"]:
            print(f"  [{ent['label']}] {ent['text']}")
    print(f"\nResponse message:\n\n{result_positive['response_message']}")
    if result_positive.get("recommendations"):
        print(f"\nRecommended trips:")
        for rec in result_positive["recommendations"]:
            activities = ", ".join(rec['extra_activities'])
            print(f"  #{rec['rank']}: {rec['city']}, {rec['country']} - {rec['cost']} EUR, {rec['days']} days, Start: {rec['start_date']}")
            print(f"       Activities: {activities}")
            print(f"       Details: {rec['details'][:100]}...")

    # ---- Test full_review_chain with a negative review ----
    negative_review = """dump, place dump, incredibly noisy windows closed not sleep,
    cut trip short days, staff extremely rude, toilet situated way legs literally
    tucked sink sit sideways, smelled urine, air conditioning worked poorly,
    rundown kind seedy town, not recommend, low point month trip italy"""

    print("\n" + "="*80)
    print("FULL CHAIN TEST 2: Negative Review (Italy)")
    print("="*80)
    print(f"Review: {negative_review.strip()[:200]}...")

    result_negative = full_review_chain.invoke({"review": negative_review})

    print(f"\nPositive? {result_negative['positive_sentiment']}")
    print(f"Reasoning: {result_negative.get('reasoning', 'N/A')}")
    if result_negative.get("entities"):
        print(f"\nEntities:")
        for ent in result_negative["entities"]:
            print(f"  [{ent['label']}] {ent['text']}")
    print(f"\nResponse message:\n\n{result_negative['response_message']}")
