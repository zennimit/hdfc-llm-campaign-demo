import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import faiss
import openai

# ——— CONFIG ———
openai.api_key = st.secrets["OPENAI_API_KEY"]
DATA_PATH = "data/transactions.csv"

# ——— UI HEADER ———
st.title("LLM-Augmented Cohort Demo for HDFC Campaigns")

# ——— LOAD & NORMALIZE DATA ———
df = pd.read_csv(DATA_PATH)

# Auto-rename common variants to user_id
if "Customer ID" in df.columns:
    df = df.rename(columns={"Customer ID": "user_id"})
elif "customer_id" in df.columns:
    df = df.rename(columns={"customer_id": "user_id"})
# Add more elifs here if your file uses a different name...

# Debug: show detected columns (remove after this works)
st.write("Detected columns:", df.columns.tolist())

# Basic sanity check
if "user_id" not in df.columns or "event_text" not in df.columns:
    st.error("CSV must contain columns: user_id, event_text")
    st.stop()

# ——— EMBEDDING & CLUSTERING HELPERS ———
@st.cache_data
def embed_texts(texts: list[str]) -> np.ndarray:
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return np.array([r["embedding"] for r in resp["data"]], dtype="float32")

@st.cache_data
def build_user_medoids(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    For each user, cluster their event embeddings and pick one medoid per cluster.
    Returns:
      - medoids: { user_id: np.ndarray[[vector, ...], ...] }
      - user_index: { user_id: idx } mapping to an integer index for FAISS
    """
    medoids = {}
    user_index = {}
    idx_counter = 0

    for uid, group in df.groupby("user_id"):
        texts = group["event_text"].tolist()
        embs = embed_texts(texts)
        # Simple 2-cluster example; tweak n_clusters as needed
        cl = AgglomerativeClustering(n_clusters=2).fit(embs)
        vectors = []
        for label in np.unique(cl.labels_):
            sub_idxs = np.where(cl.labels_ == label)[0]
            sub_embs = embs[sub_idxs]
            # pick medoid = vector with minimal avg distance
            dists = np.linalg.norm(sub_embs[:, None] - sub_embs[None, :], axis=2).mean(axis=1)
            medoid_vec = sub_embs[np.argmin(dists)]
            vectors.append(medoid_vec)
        medoids[uid] = np.stack(vectors)
        user_index[uid] = idx_counter
        idx_counter += len(vectors)

    return medoids, user_index

def build_faiss_index(medoids: dict) -> faiss.IndexFlatL2:
    dim = next(iter(medoids.values())).shape[1]
    index = faiss.IndexFlatL2(dim)
    for vectors in medoids.values():
        index.add(vectors)
    return index

def vector_search(intent_vecs: list[np.ndarray], index: faiss.IndexFlatL2,
                  user_index: dict, top_k: int = 100) -> list[str]:
    # search each intent, collect top_k per intent
    D, I = index.search(np.vstack(intent_vecs), top_k)
    # invert user_index mapping: FAISS id -> user_id
    inv_map = {v: k for k, v in user_index.items()}
    hits = set()
    for row in I:
        for fid in row:
            uid = inv_map.get(fid)
            if uid:
                hits.add(uid)
    return list(hits)

# ——— BUILD INDEX (with spinner) ———
with st.spinner("Building user clusters & FAISS index…"):
    medoids, user_index = build_user_medoids(df)
    faiss_index = build_faiss_index(medoids)
st.success(f"Indexed {len(user_index)} users")

# ——— CAMPAIGN UI ———
st.header("Run a Campaign")
goal_text = st.text_input(
    "Describe your campaign goal in plain English",
    value="Weekend getaway for flight & hotel bookers"
)

if st.button("Generate Cohort"):
    with st.spinner("Generating intent vectors…"):
        # Example: derive two intents from the goal
        intent_prompts = [
            f"{goal_text} — identify flight-bookers",
            f"{goal_text} — identify hotel-stayers"
        ]
        intent_vecs = [embed_texts([p])[0] for p in intent_prompts]

    with st.spinner("Searching for matching users…"):
        cohort = vector_search(intent_vecs, faiss_index, user_index, top_k=200)

    st.success(f"🎯 Target cohort: {len(cohort)} users")
    st.write(cohort)

