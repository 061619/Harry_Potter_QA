<b>Harry Potter Q&A Assistant</b>
<br><br>
<b>Overview</b>
<br><br>
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about the book Harry Potter and the Prisoner of Azkaban. The system extracts text from a PDF, processes it into smaller coherent chunks, converts these chunks into dense vector embeddings, stores them in a FAISS index for efficient retrieval, and utilizes the Gemini API to generate responses based on retrieved context.
<br><br>
<b>Features</b>
<br><br>
Extracts text from a PDF using pdfplumber.
<br><br>
Chunks the text into smaller segments to improve retrieval efficiency.
<br><br>
Generates dense vector representations using Sentence Transformers.
<br><br>
Implements FAISS (Facebook AI Similarity Search) for efficient nearest-neighbor search.
<br><br>
Queries the Gemini API to generate answers based on the retrieved chunks.
<br><br>
Provides an interactive command-line interface where users can ask questions about the book.
<br><br>
<b>Dependencies</b>
<br><br>
To run this project, install the required dependencies:
<br><br>
pip install pdfplumber sentence-transformers faiss-cpu numpy pydantic python-dotenv requests
<br><br>
<b>How It Works</b>
<br><br>
1. PDF Parsing
<br><br>
The project begins by extracting text from the provided PDF book using pdfplumber. The function parse_pdf(file_path) reads the entire text from the PDF document.
<br><br>
2. Text Chunking and Preprocessing
<br><br>
The extracted text is then split into smaller overlapping chunks for better semantic clarity. This improves retrieval efficiency and context retention when querying the system.
<br><br>
3. Creating Embeddings
<br><br>
Each chunk is transformed into a dense vector representation using all-MiniLM-L6-v2, a model from Sentence Transformers. These embeddings enable semantic search over the text.
<br><br>
from sentence_transformers import SentenceTransformer
<br><br>
4. FAISS Indexing
<br><br>
FAISS (Facebook AI Similarity Search) is used to store and quickly retrieve the most relevant text chunks based on user queries.
<br><br>
import faiss<br>
import numpy as np
<br><br>
5. Querying Gemini API
<br><br>
When a user asks a question, the system retrieves the top 5 most relevant text chunks using FAISS and then sends this combined context to Gemini API for answer generation.
<br><br>
import requests<br>
import json
<br><br>
6. End-to-End Query Handling
<br><br>
When a user inputs a question, its embedding is computed, the nearest chunks are retrieved, and the Gemini API is queried to generate an answer.
<br><br>

7. Command-Line Interface (CLI)
<br><br>
The project includes an interactive CLI where users can ask questions and receive responses in real time.
<br><br>
API Key Configuration
<br><br>
The Gemini API key should be stored securely using a .env file:
<br><br>
GEMINI_API_KEY=your_actual_api_key_here
<br>
And loaded using dotenv:
<br>
from dotenv import load_dotenv<br>
import os<br>

load_dotenv()<br>
API_KEY = os.getenv("GEMINI_API_KEY")
<br><br>
<b>Future Improvements</b>
<br><br>
Implement multi-document retrieval to support multiple books.
<br><br>
Optimize FAISS retrieval with Hierarchical Navigable Small World (HNSW) indexing.
<br><br>
Use better chunking strategies, such as semantic segmentation instead of fixed-size chunks.
<br><br>
Replace the Gemini API with open-source LLMs for cost efficiency.
<br><br>
Build a web-based UI using Streamlit for easier interaction.
<br><br>
<b>License</b>
<br><br>
This project is open-source and available under the MIT License.
<br><br>
<b>Acknowledgments</b>
<br><br>
https://python.langchain.com/docs/integrations/document_loaders/pdfplumber/ for PDF text extraction.
<br><br>
https://sbert.net/ for generating embeddings.
<br><br>
https://ai.meta.com/tools/faiss/ for efficient vector search.
<br><br>
https://ai.google.dev/ for text generation.
