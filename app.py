import io
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ——— CONFIG —————————————————————————————————————————————————————————————————
openai.api_key = st.secrets["OPENAI_API_KEY"]
DATA_PATH = "data/transactions.csv"
NUM_GLOBAL_CLUSTERS = 5   # step 2: how many “always-on” cohorts

# ——— 1) INGEST ALL CUSTOMER DATA ——————————————————————————————————————————
st.title("LLM-Powered Campaign Engine Prototype")
st.markdown("## Step 1: Load & inspect data")
df = pd.read_csv(DATA_PATH)

# normalize column names
if "Customer ID" in df.columns:
    df = df.rename(columns={"Customer ID": "user_id"})
elif "customer_id" in df.columns:
    df = df.rename(columns={"customer_id": "user_id"})

# build event_text
df["event_text"] = (
    df["Merchant"] + " | " + df["Category"] + " | ₹" + df["Amount (INR)"].astype(str)
)
st.write("Sample rows:")
st.dataframe(df.head())

# ——— 2) AUTO-CREATE ALWAYS-ON GLOBAL CLUSTERS ——————————————————————————————
st.markdown("## Step 2: Build global clusters")
@st.cache_data
def build_global_clusters(df, k, samples_per_user=5):
    # 1) Sample up to `samples_per_user` events per user (most recent or random)
    #    Here we pick the most recent N events for each user
    df_sorted = df.sort_values("Timestamp")
    sampled = (
        df_sorted
        .groupby("user_id")
        .tail(samples_per_user)
        .reset_index(drop=True)
    )

    # 2) Embed those sampled event_texts in one batch
    texts = sampled["event_text"].tolist()
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embs = np.array([d["embedding"] for d in resp["data"]], dtype="float32")
    sampled["_emb"] = list(embs)

    # 3) Aggregate sampled embeddings into one user vector (mean of up to N samples)
    user_vecs = (
        sampled
        .groupby("user_id")["_emb"]
        .apply(lambda vs: np.mean(vs.tolist(), axis=0))
    )
    users = user_vecs.index.tolist()
    matrix = np.vstack(user_vecs.values)

    # 4) KMeans clustering on the per-user aggregated vectors
    km = KMeans(n_clusters=k, random_state=42).fit(matrix)
    labels = km.labels_
    centroids = km.cluster_centers_

    return users, matrix, labels, centroids

with st.spinner("Clustering users into global cohorts…"):
    users, user_matrix, user_labels, cluster_centroids = build_global_clusters(df, NUM_GLOBAL_CLUSTERS)
cluster_df = pd.DataFrame({
    "user_id": users,
    "cluster": user_labels
})
st.write(cluster_df["cluster"].value_counts())

# ——— 2a) LLM-GENERATE CLUSTER RATIONALES/NAMES ——————————————————————————
st.markdown("### Cluster names & rationales (auto-generated)")
@st.cache_data
def name_and_rationalize_clusters(centroids, k):
    names, rationales = [], []
    for i in range(k):
        # For each centroid, we ask the LLM to name & describe
        prompt = f"""
Here is a cluster centroid in embedding space (vector omitted for brevity). 
Based on the kinds of purchase events we've seen, suggest:
1) A short name for this cluster (e.g. "Weekend Shoppers")
2) A one-sentence rationale describing their common behavior.
"""
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a marketing analyst."},
                      {"role":"user","content":prompt}],
            temperature=0.7
        )
        out = resp.choices[0].message.content.strip().split("\n",1)
        names.append(out[0].strip())
        rationales.append(out[1].strip() if len(out)>1 else "")
    return names, rationales

cluster_names, cluster_rats = name_and_rationalize_clusters(cluster_centroids, NUM_GLOBAL_CLUSTERS)
for i,(n,r) in enumerate(zip(cluster_names, cluster_rats)):
    st.markdown(f"**Cluster {i} — {n}**: {r}")

# ——— 3) INPUT CAMPAIGN GOAL —————————————————————————————————————————————
st.markdown("## Step 3: Define your campaign goal")
campaign_goal = st.text_input("Campaign goal (natural language)", 
                              "Weekend getaway for flight & hotel bookers")

# ——— 4) IDENTIFY USER SEGMENTS (match goal → clusters) —————————————————————————
st.markdown("## Step 4: Select target clusters for this goal")
@st.cache_data
def rank_clusters_for_goal(goal, centroids):
    # embed the goal
    gv = openai.Embedding.create(model="text-embedding-ada-002", input=[goal])["data"][0]["embedding"]
    sims = cosine_similarity([gv], centroids)[0]
    # return cluster indices sorted by descending similarity
    return list(np.argsort(sims)[::-1]), sims

ranked_clusters, cluster_sims = rank_clusters_for_goal(campaign_goal, cluster_centroids)
st.write("Clusters ranked by relevance:", ranked_clusters[:3], 
         "with scores", np.round(cluster_sims[:3],3))

# ——— 5) INPUT CAMPAIGN TYPES + COSTS —————————————————————————————————————
st.markdown("## Step 5: Define your available campaigns & costs")
# expect input as CSV in the text_area: campaign_id,description,cost
campaign_csv = st.text_area("Paste CSV: campaign_id,description,cost", 
 """campaign_id,description,cost
A,10% off flight booking,100
B,₹500 cashback on hotel,80
C,Buy-1-Get-1 ride voucher,60""")
camp_df = pd.read_csv(io.StringIO(campaign_csv))
st.dataframe(camp_df)

# ——— 6) ASSIGN BEST CAMPAIGN PER USER —————————————————————————————————————
st.markdown("## Step 6: Assign best campaign to each user (propensity via embedding sim)")
@st.cache_data
def compute_propensities(df, camp_df):
    # embed campaign descriptions
    descs = camp_df["description"].tolist()
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=descs)
    camp_embs = np.array([d["embedding"] for d in resp["data"]], dtype="float32")
    # reuse user_matrix from step 2
    # propensity = cosine sim between each user_vec and each campaign_emb
    sims = cosine_similarity(user_matrix, camp_embs)
    # build a table
    rows = []
    for ui, uid in enumerate(users):
        best_j = np.argmax(sims[ui])
        rows.append({
            "user_id": uid,
            "campaign_id": camp_df.loc[best_j,"campaign_id"],
            "propensity": float(sims[ui,best_j]),
            "cost": float(camp_df.loc[best_j,"cost"])
        })
    return pd.DataFrame(rows)

assign_df = compute_propensities(df, camp_df)
st.dataframe(assign_df.head())

# ——— 7) INPUT TOTAL BUDGET —————————————————————————————————————————————
st.markdown("## Step 7: Define your total budget")
total_budget = st.number_input("Total budget (₹)", min_value=0, value=100000)

# ——— 8) BUDGET-CONSTRAINED SELECTION —————————————————————————————————————
st.markdown("## Step 8: Limit assignments under budget (greedy)")
def budget_constrained(df, B):
     # compute propensity-per-cost ratio, then sort descending by it
     df = df.copy()
     df["roi"] = df["propensity"] / df["cost"]
     df = df.sort_values(by="roi", ascending=False)

    spent = 0
    keep = []
    for _, row in df.iterrows():
        if spent + row["cost"] <= B:
            keep.append(True)
            spent += row["cost"]
        else:
            keep.append(False)
    df["selected"] = keep
    return df

final_df = budget_constrained(assign_df, total_budget)
st.write("Total spent (approx):", int(final_df.loc[final_df["selected"],"cost"].sum()))
st.dataframe(final_df[final_df["selected"]].head())

# ——— 9) DISPLAY FINAL TABLE WITH RATIONALES ——————————————————————————————
st.markdown("## Step 9: Final cohorts with LLM rationales")
selected = final_df[final_df["selected"]]
for cluster_idx in ranked_clusters[:2]:  # just top-2 clusters for brevity
    seg_users = cluster_df.loc[cluster_df["cluster"]==cluster_idx,"user_id"]
    seg_selected = selected[selected["user_id"].isin(seg_users)]
    if seg_selected.empty: continue

    st.markdown(f"### Cohort {cluster_idx}: {cluster_names[cluster_idx]}")
    # ask LLM for cohort rationale once
    rationale = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a campaign reasoning assistant."},
            {"role":"user","content":
f"""Campaign Goal: {campaign_goal}
Cohort: {cluster_names[cluster_idx]}
Users: {seg_selected['user_id'].tolist()}

Write a one-sentence rationale why this cohort matches the goal."""
            }
        ]
    ).choices[0].message.content.strip()
    st.write("**Cohort Rationale:**", rationale)

    # now per-user rationale
    rows = []
    for _, row in seg_selected.iterrows():
        usr, camp = row["user_id"], row["campaign_id"]
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a campaign reasoning assistant."},
                {"role":"user","content":
f"""User ID: {usr}
User events: {df[df['user_id']==usr]['event_text'].tolist()}
Assigned Campaign: {camp} - {camp_df.loc[camp_df['campaign_id']==camp,'description'].iloc[0]}

Write a one-line rationale why we picked this campaign for this user."""
                }
            ]
        )
        rows.append((usr, camp, chat.choices[0].message.content.strip()))

    # display as table
    result_tbl = pd.DataFrame(rows, columns=["user_id","campaign_id","rationale"])
    st.table(result_tbl)
