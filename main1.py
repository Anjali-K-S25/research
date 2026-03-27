import streamlit as st
import pandas as pd
import io

from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gemini_file import ask_gemini

import plotly.graph_objects as go
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")

# ===============================
# 🎨 THEME CONTROLS (ENHANCED)
# ===============================
st.sidebar.header("Theme Settings")

theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])

if theme_mode == "Light":
    bg_color = "#f5f7fb"
    primary_color = "#1e40af"     # strong blue
    secondary_color = "#2563eb"
    text_color = "#111827"
    card_bg = "#ffffff"
    border_color = "#d1d5db"
    button_text = "#ffffff"

else:
    bg_color = "#0b1220"
    primary_color = "#38bdf8"     # bright cyan
    secondary_color = "#0ea5e9"
    text_color = "#e5e7eb"
    card_bg = "#111827"
    border_color = "#334155"
    button_text = "#ffffff"

# ===============================
# GLOBAL STYLE (FIXED + STRONG)
# ===============================
st.markdown(f"""
<style>

/* APP BACKGROUND */
.stApp {{
    background-color: {bg_color} !important;
    color: {text_color};
    font-family: 'Inter', sans-serif;
}}

/* BUTTONS (FIXED — NO WHITE ISSUE) */
.stButton button {{
    background-color: {primary_color};
    color: {button_text};
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}}

.stButton button:hover {{
    background-color: {secondary_color};
    color: {button_text};
}}

.stButton button:active {{
    background-color: {secondary_color} !important;
    transform: scale(0.97);
    color: {button_text} !important;
}}

.stButton button:focus {{
    background-color: {primary_color} !important;
    color: {button_text} !important;
    outline: none;
}}

/* INPUT */
.stTextInput input {{
    border-radius: 8px;
    border: 1px solid {border_color};
    background-color: {card_bg};
    color: {text_color};
    padding: 8px;
}}

/* METRICS */
[data-testid="metric-container"] {{
    background-color: {card_bg};
    border-radius: 12px;
    padding: 14px;
    border: 1px solid {border_color};
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}}

/* TABLE */
.stDataFrame {{
    background-color: {card_bg};
    border-radius: 10px;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background-color: {card_bg};
    border-right: 1px solid {border_color};
}}

/* EXPANDER */
.streamlit-expanderHeader {{
    color: {primary_color};
    font-weight: 600;
}}

/* TEXT IMPROVEMENT */
.stApp {{
    line-height: 1.6;
}}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER (CLEAN + STRONG VISIBILITY)
# ===============================
st.markdown(f"""
<div style="
    text-align: center;
    padding: 22px;
    border-radius: 14px;
    background: linear-gradient(135deg, {primary_color}, {secondary_color});
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
">

<h1 style="
    margin: 0;
    font-size: 40px;
    font-weight: 800;
">

AI-Powered Research Paper Analysis System

</h1>

<p style="
    margin-top: 8px;
    font-size: 14px;
    opacity: 0.95;
">
Query research papers and explore structured knowledge graphs
</p>

</div>
""", unsafe_allow_html=True)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["Research Paper QA", "Knowledge Graph Explorer"])

# Tab 1-> RAG Research Paper Question Answering 

with tab1:
    st.markdown("""
    <style>
    .stTextInput input {
        background-color: #F5F5F5;
        color: #000000;
        border-radius: 10px;
        border: 2px solid #00FFFF;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_vector_db():

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "research_papers_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_db


    vector_db = load_vector_db()

    # Show total papers
    st.write("Total Papers in Database:", vector_db.index.ntotal)

    # User input
    user_query = st.text_input(" Ask a question about research papers:")

    # Search button
    if st.button("Search"):

        results = vector_db.similarity_search(user_query, k=3)

        content = ""

        for idx, doc in enumerate(results, 1):

            title = doc.metadata.get("title", f"Paper {idx}")

            content += f"""
            Paper Title: {title}

            Paper Content:
            {doc.page_content}

            """

        print("whole content",content )
        # Call Gemini
        with st.spinner(" Analyzing research papers and generating insights..."):
            response = ask_gemini(content, user_query)
        print("gemini response", response)
        # -------------------------
        # Extract Answer and Paper
        # -------------------------

        answer = ""
        paper_titles = []

        if "Research Paper:" in response:
            parts = response.split("Research Paper:")
            
            answer = parts[0].replace("Answer:", "").strip()

            papers_text = parts[1].strip()

            # Split multiple papers by comma
            paper_titles = [p.strip() for p in papers_text.split(",")]

        else:
            answer = response.strip()

        # Show AI answer
        st.subheader("🤖 AI Generated Insight")
        st.write(answer)

        # -------------------------
        # Show Only Relevant Paper
        # -------------------------

        if paper_titles and "none" not in [p.lower() for p in paper_titles]:

            st.subheader("📄 Relevant Research Papers")

            for doc in results:

                title = doc.metadata.get("title", "")

                for p in paper_titles:

                    if title.lower() == p.lower():

                        with st.expander(f"📄 {title}"):

                            st.write(doc.page_content)

        else:
            st.warning("No relevant research paper found.")
 
# ===============================
# TAB 2
# ===============================
with tab2:

    st.subheader("Knowledge Graph Explorer")

    # ---------------------------
    # Neo4j Driver
    # ---------------------------
    @st.cache_resource
    def get_driver():
        return GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=('neo4j','info@1234')
        )

    driver = get_driver()

    # ---------------------------
    # Fetch Domains
    # ---------------------------
    @st.cache_data
    def get_domain():
        query = "MATCH (d:Domain) RETURN d.name as domain"
        with driver.session() as session:
            result = session.run(query)
            return sorted(set([r["domain"] for r in result]))

    domain = st.selectbox("Select Research Domain", get_domain())

    # ---------------------------
    # Fetch Graph Data
    # ---------------------------
    def get_graph_data(domain):
        query = """
        MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
        WHERE toLower(d.name) = toLower($domain)

        OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
        OPTIONAL MATCH (p)-[:USES]-(m:Method)
        OPTIONAL MATCH (p)-[:EVALUATED_BY]->(me:Metric)

        RETURN p.title AS paper,
               a.name AS author,
               m.name AS method,
               me.name AS metric,
               d.name AS domain
        """

        with driver.session() as session:
            result = session.run(query, domain=domain)
            return [r.data() for r in result]

    # ---------------------------
    # Neo4j Graph (PyVis)
    # ---------------------------
    def draw_graph(data, theme):

        bg = "#ffffff" if theme == "Light" else "#0b1220"
        font = "#222222" if theme == "Light" else "#ffffff"

        net = Network(height="600px", width="100%", bgcolor=bg, font_color=font)
        net.repulsion(node_distance=180, spring_length=120)

        for row in data:
            p = row['paper']
            a = row['author']
            m = row['method']
            me = row.get('metric')
            d = row['domain']

            net.add_node(p, label=p, color="#1f4e79", size=22)

            if a:
                net.add_node(a, label=a, color="#2a9d8f", size=12)
                net.add_edge(a, p)

            if m:
                net.add_node(m, label=m, color="#f4a261", size=12)
                net.add_edge(p, m)

            if me:
                net.add_node(me, label=me, color="#e63946", size=12)
                net.add_edge(p, me)

            if d:
                net.add_node(d, label=d, color="#6a4c93", size=16)
                net.add_edge(p, d)

        net.save_graph("graph.html")

        with open("graph.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)

    # ---------------------------
    # Plotly 3D Rotating Graph
    # ---------------------------
    def draw_plotly_3d_graph(data):

        if not data:
            st.warning("No data for 3D graph")
            return

        nodes = {}
        edges = []

        def add_node(name, t, color):
            if name and name not in nodes:
                nodes[name] = {"type": t, "color": color}

        for row in data:
            p = row.get('paper')
            a = row.get('author')
            m = row.get('method')
            me = row.get('metric')
            d = row.get('domain')

            add_node(p, "paper", "#1f4e79")

            if a:
                add_node(a, "author", "#2a9d8f")
                edges.append((a, p))

            if m:
                add_node(m, "method", "#f4a261")
                edges.append((p, m))

            if me:
                add_node(me, "metric", "#e63946")
                edges.append((p, me))

            if d:
                add_node(d, "domain", "#6a4c93")
                edges.append((p, d))

        node_list = list(nodes.keys())
        N = len(node_list)

        angles = np.linspace(0, 2*np.pi, N)
        x = 3 * np.cos(angles)
        y = 3 * np.sin(angles)

        z = []
        for n in node_list:
            t = nodes[n]["type"]
            z.append({"paper":0,"author":1,"method":-1,"metric":2}.get(t,-2))

        pos = np.column_stack((x, y, z))
        idx = {n:i for i,n in enumerate(node_list)}

        def create_frame(p):
            edge_x, edge_y, edge_z = [], [], []

            for s,t in edges:
                if s in idx and t in idx:
                    i,j = idx[s], idx[t]
                    edge_x += [p[i][0], p[j][0], None]
                    edge_y += [p[i][1], p[j][1], None]
                    edge_z += [p[i][2], p[j][2], None]

            edge_trace = go.Scatter3d(
                x=edge_x,y=edge_y,z=edge_z,
                mode='lines',
                line=dict(width=2,color='#888')
            )

            node_trace = go.Scatter3d(
                x=p[:,0],y=p[:,1],z=p[:,2],
                mode='markers+text',
                text=node_list,
                marker=dict(size=6,
                            color=[nodes[n]["color"] for n in node_list])
            )

            return [edge_trace,node_trace]

        frames = []
        for t in range(40):
            angle = t * 0.1
            rot = np.array([
                [np.cos(angle),-np.sin(angle),0],
                [np.sin(angle), np.cos(angle),0],
                [0,0,1]
            ])
            frames.append(go.Frame(data=create_frame(pos @ rot)))

        fig = go.Figure(data=create_frame(pos), frames=frames)

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="#0b1220"
            ),
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="▶ Play",
                              method="animate",
                              args=[None])]
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # MAIN UI
    # ---------------------------
    if domain:

        data = get_graph_data(domain)

        if not data:
            st.warning("No data available")
        else:
            df = pd.DataFrame(data)

            # Metrics
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Papers", df['paper'].nunique())
            col2.metric("Authors", df['author'].nunique())
            col3.metric("Methods", df['method'].nunique())
            col4.metric("Metrics", df['metric'].nunique())

            st.markdown("---")

            # Dataset
            st.subheader("Research Dataset")
            st.dataframe(df, use_container_width=True)

            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            buf.seek(0)

            st.download_button("Download Dataset", buf,
                               file_name=f"{domain}.xlsx")

            st.markdown("---")

            # Legend
            st.markdown("""
**Legend:**
- Papers → Dark Blue  
- Authors → Teal  
- Methods → Orange  
- Metrics → Red  
- Domain → Purple  
""")

            theme = st.radio("Graph Theme", ["Light","Dark"], horizontal=True)

            st.markdown("### 🔷 Neo4j Graph")
            draw_graph(data, theme)

            st.markdown("---")

            st.markdown("### 🔶 3D Rotating Graph")
            draw_plotly_3d_graph(data)