# AI‑Powered Research Paper Summarizer & Insight Extractor

This project processes research papers to extract summaries, insights, and visualizations. It can parse scientific documents, build searchable indexes, and display results in meaningful formats including 2D and 3D plots.

## Repository Overview
 - **data/** – Input PDFs and source files.  
- **parsed_output/** – Processed text, summaries, and insights of PDFs.  
- **research_papers_faiss/** – Vector indexes and embeddings for semantic search.  
- **arxiv_papers/** – Papers fetched from arXiv.  
- **data_inject/** – Scripts or files for injecting data into the pipeline.  
- **extract_pdf/** – Utilities for parsing PDF content.  
- **gemini_file/** – Files related to Gemini/Groq summarization or processing.  
- **graph/** – Knowledge graph files and scripts.  
- **helper_function.py** – Utility functions for parsing, summarization, and embeddings.  
- **main1.py** – Main script and user interface.  
- **ployly_graph/** – Converting the data to 3D.  
- **pubmed/** – Scripts and data for PubMed paper processing.  
- **pubmed_multiple_queries/** – Output of pubmed.  
- **requirements.txt** – Project dependencies.  
- **upload_on_neo4j.py** – Upload extracted insights to Neo4j.  


## Key Features

✔ Parses and indexes research papers  
✔ Generates summaries using AI models  
✔ Visualizes data and insights in 2D and 3D graphs  
✔ Supports exporting results (Excel/CSV)  
✔ (Optional) Uploads knowledge graph data to Neo4j  
