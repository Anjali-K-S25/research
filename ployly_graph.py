import json
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# LOAD FAISS + EMBEDDINGS
# ---------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "research_papers_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

docs = vector_db.docstore._dict

# ---------------------------
# EXTRACT TEXT + METADATA
# ---------------------------
texts = []
titles = []
domains = []

for doc_id, doc in docs.items():
    texts.append(doc.page_content)
    titles.append(doc.metadata.get("title", "No Title"))
    domains.append(", ".join(doc.metadata.get("domain", [])) if doc.metadata.get("domain") else "Unknown")

# ---------------------------
# CREATE EMBEDDINGS MATRIX
# ---------------------------
emb_matrix = embeddings.embed_documents(texts)
emb_matrix = np.array(emb_matrix)

# ---------------------------
# CREATE SIMILARITY MATRIX
# ---------------------------
similarity_matrix = cosine_similarity(emb_matrix)

# ---------------------------
# CREATE EDGES (THRESHOLD)
# ---------------------------
edges = []
threshold = 0.7  # adjust

for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        if similarity_matrix[i][j] > threshold:
            edges.append((i, j, similarity_matrix[i][j]))

# ---------------------------
# NODE POSITIONS (3D)
# ---------------------------
# reduce dimensions to 3D
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
coords = pca.fit_transform(emb_matrix)

x, y, z = coords[:,0], coords[:,1], coords[:,2]

# ---------------------------
# CREATE EDGE TRACE
# ---------------------------
edge_x = []
edge_y = []
edge_z = []

for i, j, sim in edges:
    edge_x += [x[i], x[j], None]
    edge_y += [y[i], y[j], None]
    edge_z += [z[i], z[j], None]

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    line=dict(width=1),
    hoverinfo='none',
    mode='lines'
)

# ---------------------------
# CREATE NODE TRACE
# ---------------------------
node_trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    text=titles,
    hovertext=[
        f"Title: {titles[i]}<br>Domain: {domains[i]}"
        for i in range(len(titles))
    ],
    marker=dict(
        size=6,
    )
)

# ---------------------------
# PLOT
# ---------------------------
fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    title="Research Paper Knowledge Graph (3D)",
    showlegend=False,
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()