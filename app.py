import streamlit as st
import pandas as pd
import faiss
from sklearn.cluster import AgglomerativeClustering
import openai
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

@st.cache_data
def embed_text(texts):
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
    return np.array([r["embedding"] for r in resp["data"]])

@st.cache_data
def build_user_medoids(df):
    medoids, user_index = {}, {}
    for uid, group in df.groupby("user_id"):
        embs = embed_text(group["event_text"].tolist())
        # cluster into 2 clusters (for simplicity)
        cl = AgglomerativeClustering(n_clusters=2).fit(embs)
        # pick the first medoid per cluster
        medoid_idxs = []
        for label in np.unique(cl.labels_):
            idxs = np.where(cl.labels_ == label)[0]
            # medoid = point with min avg distance
            sub = embs[idxs]
            distances = np.linalg.norm(sub[:,None] - sub[None,:], axis=2).mean(axis=1)
            medoid_idxs.append(idxs[np.argmin(distances)])
        medoids[uid] = embs[medoid_idxs]
        user_index[uid] = len(user_index)
    return medoids, user_index

# build FAISS index
def build_index(medoids, user_index):
    dim = next(iter(medoids.values())).shape[1]
    index = faiss.IndexFlatL2(dim)
    for uid, vectors in medoids.items():
        for vec in vectors:
            index.add(vec[np.newaxis])
    return index

def search_users(intent_vecs, index, user_index, top_k=50):
    D, I = index.search(np.vstack(intent_vecs), top_k)
    # map back to user_ids (simplest: ignore duplicates)
    inv = {v:k for k,v in user_index.items()}
    results = set(inv[i] for i in I.flatten() if i in inv)
    return list(results)

st.title("LLM-Augmented Campaign Prototype")

df = load_data()
st.sidebar.write("### Data Preview"); st.sidebar.dataframe(df.head())

# 1) Build medoids & index
medoids, user_index = build_user_medoids(df)
index = build_index(medoids, user_index)

# 2) Define campaign goal
goal = st.text_input("Enter Campaign Goal", 
    "Weekend getaway for flight & hotel bookers")
if st.button("Run Campaign"):
    # 3) Have LLM output intent vectors via embedding
    #    (in production, you'd prompt for NL intents and embed them)
    intents = [
      openai.Embedding.create(model="text-embedding-ada-002", 
        input=f"{goal} — identify flight-bookers")["data"][0]["embedding"],
      openai.Embedding.create(model="text-embedding-ada-002", 
        input=f"{goal} — identify hotel-stayers")["data"][0]["embedding"]
    ]
    # 4) Vector search
    users = search_users(intents, index, user_index, top_k=100)
    st.success(f"Target cohort: {len(users)} users")
    st.write(users)
