import streamlit as st
import importlib
import sys
import os

# Add scripts/ to Python path so importlib can find the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# RUN: streamlit run scripts/app.py

@st.cache_resource
def load_recommender():
    """Import ner-trip-recommender module once and cache it."""
    return importlib.import_module("ner-trip-recommender")


recommender = load_recommender()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Trip Review Analyzer", page_icon="🌍", layout="wide")

# Custom button color
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #4FC3F7;
        border-color: #4FC3F7;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #29B6F6;
        border-color: #29B6F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Trip Review Analyzer")
st.markdown("*ML + NER + Sentiment + RAG + AI (LLM) + Agent (Router) — enter a review and the system will detect sentiment and suggest a response.*")

st.divider()

# ---------------------------------------------------------------------------
# Example reviews
# ---------------------------------------------------------------------------
EXAMPLES = {
    "-- select an example --": "",
    "Positive (Egypt)": (
        "amazing beach resort red sea, snorkeling was incredible, saw beautiful coral reef "
        "and colorful fish, staff very friendly, food was great, egyptian cuisine delicious, "
        "visited pyramids on day trip to cairo, unforgettable experience in egypt"
    ),
    "Positive (Spain)": (
        "hotel america nice hotel good location stayed 3 nights hotel america late december, "
        "rooms modern nice, really liked location hotel, located 3 blocks main area, excellent location "
        "base stay explore interesting parts city, able walk las ramblas neighborhoods gothic district, "
        "walked sacred familia cathedral no 15 minutes morning, typical spain hard fault hotel america"
    ),
    "Negative (Italy)": (
        "dump, place dump, incredibly noisy windows closed not sleep, cut trip short days, "
        "staff extremely rude, toilet situated way legs literally tucked sink sit sideways, "
        "smelled urine, air conditioning worked poorly, rundown kind seedy town, "
        "not recommend, low point month trip italy"
    ),
}

selected = st.selectbox("Example reviews:", list(EXAMPLES.keys()))

review_text = st.text_area(
    "Enter customer review:",
    value=EXAMPLES[selected],
    height=160,
    placeholder="The hotel was amazing, great location near the beach...",
)

# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
if st.button("Analyze review", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Analyzing sentiment and generating response..."):
        result = recommender.full_review_chain.invoke({"review": review_text})

    sentiment_score = result.get("sentiment_score")

    # ---- Sentiment badge ----
    st.divider()
    if sentiment_score:
        if sentiment_score >= 4:
            st.success(f"**Sentiment Score: {sentiment_score} / 5** ✅ (Positive)")
        elif sentiment_score == 3:
            st.info(f"**Sentiment Score: {sentiment_score} / 5** ➖ (Neutral)")
        else:
            st.error(f"**Sentiment Score: {sentiment_score} / 5** ❌ (Negative)")
    else:
        st.warning("**Sentiment: Unable to determine**")

    if result.get("reasoning"):
        st.caption(f"**Reasoning:** {result['reasoning']}")

    # ---- NER entities (positive branch) ----
    entities = result.get("entities", [])
    if entities:
        st.divider()
        st.subheader("Recognized Entities (NER)")

        label_colors = {
            "geo": "#4FC3F7",
            "gpe": "#81C784",
            "nat": "#FFB74D",
        }
        label_names = {
            "geo": "Location",
            "gpe": "Country / City",
            "nat": "Natural landmark",
        }

        pills_html = " ".join(
            f'<span style="display:inline-block;padding:4px 12px;margin:3px;'
            f'border-radius:16px;font-size:14px;'
            f'background-color:{label_colors.get(ent["label"], "#90A4AE")};'
            f'color:white;">'
            f'{ent["text"]} <small>({label_names.get(ent["label"], ent["label"])})</small>'
            f'</span>'
            for ent in entities
        )
        st.markdown(pills_html, unsafe_allow_html=True)

    # ---- Response message ----
    st.divider()
    st.subheader("Customer Response")
    st.markdown(result["response_message"])

    # ---- Recommendations (positive only) ----
    recommendations = result.get("recommendations", [])
    if recommendations:
        st.divider()
        st.subheader("Recommended Trips")

        for rec in recommendations:
            score = 1 - rec["distance"]
            activities = ", ".join(rec["extra_activities"])

            with st.container(border=True):
                col_rank, col_info = st.columns([1, 5])

                with col_rank:
                    st.metric(label="Rank", value=f"#{rec['rank']}")
                    st.caption(f"Score: {score:.2f}")

                with col_info:
                    st.markdown(f"### {rec['city']}, {rec['country']}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", f"{rec['cost']} EUR")
                    c2.metric("Days", rec["days"])
                    c3.metric("Start", rec["start_date"])

                    st.markdown(f"**Activities:** {activities}")
                    st.markdown(f"**Details:** {rec['details'][:200]}…")
