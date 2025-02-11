<b>Harry Potter Q&A Assistant</b>
<br><br>
<b>Overview</b>
<br><br>
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about the book Harry Potter and the Prisoner of Azkaban. The system extracts text from a PDF, processes it into smaller coherent chunks, converts these chunks into dense vector embeddings, stores them in a FAISS index for efficient retrieval, and utilizes the Gemini API to generate responses based on retrieved context.
<br><br>
<b>Features</b>
<br>
Extracts text from a PDF using pdfplumber.
<br>
Chunks the text into smaller segments to improve retrieval efficiency.
<br>
Generates dense vector representations using Sentence Transformers.
<br>
Implements FAISS (Facebook AI Similarity Search) for efficient nearest-neighbor search.
<br>
Queries the Gemini API to generate answers based on the retrieved chunks.
<br>
Provides an interactive command-line interface where users can ask questions about the book.
<br><br>
<b>Dependencies</b>
<br>
To run this project, install the required dependencies:
<br>
pip install pdfplumber sentence-transformers faiss-cpu numpy pydantic python-dotenv requests
<br><br>
<b>How It Works</b>
<br><br>
1. PDF Parsing
<br>
The project begins by extracting text from the provided PDF book using pdfplumber. The function parse_pdf(file_path) reads the entire text from the PDF document. PDF files often contain complex layouts, headers, footers, and multi-column text structures that can make text extraction difficult. pdfplumber is designed to handle such complexities and extract text in a structured manner.The function iterates over each page, concatenating the extracted text into a single string, ensuring that all content is captured for subsequent processing.
<br><br>
2. Text Chunking and Preprocessing
<br>
The extracted text is then split into smaller overlapping chunks for better semantic clarity. When dealing with long-form text, direct retrieval can be inefficient, as querying an entire book would result in vague or incorrect matches. Chunking ensures that each section retains contextual meaning while being small enough for effective retrieval.Overlapping chunks help maintain sentence continuity across boundaries, preventing loss of context. This is particularly useful for vector-based retrieval, where smaller, meaningful segments improve the accuracy of semantic search.
<br><br>
3. Creating Embeddings
<br>
Each chunk is transformed into a dense vector representation using all-MiniLM-L6-v2, a pre-trained model from Sentence Transformers. These embeddings convert natural language into high-dimensional numerical representations, making it easier to compare and retrieve semantically similar content.
Embeddings capture contextual meaning, allowing queries to be matched with relevant text segments, even if they do not share exact words. This enhances the system’s ability to understand user questions and retrieve the most relevant information.
<br><br>
from sentence_transformers import SentenceTransformer
<br><br>
4. FAISS Indexing
<br>
FAISS (Facebook AI Similarity Search) is used to store and quickly retrieve the most relevant text chunks based on user queries. FAISS is optimized for large-scale vector search and enables efficient nearest-neighbor retrieval.By indexing chunk embeddings, the system can rapidly find text segments that closely match the user’s question. This significantly improves search speed and accuracy compared to traditional keyword-based search.
<br><br>
import faiss<br>
import numpy as np
<br><br>
5. Querying Gemini API
<br>
When a user asks a question, the system retrieves the top 5 most relevant text chunks using FAISS and then sends this combined context to Gemini API for answer generation.The Gemini API, developed by Google, leverages large language models (LLMs) to generate human-like responses based on provided context. By supplying retrieved book chunks as input, the API can generate answers that remain faithful to the source material while maintaining coherence.
<br><br>
import requests<br>
import json
<br><br>
6. End-to-End Query Handling
<br>
When a user inputs a question, its embedding is computed, the nearest chunks are retrieved, and the Gemini API is queried to generate an answer. The combined approach of vector retrieval and generative AI ensures that responses are both contextually relevant and coherent.
<br><br>

7. Command-Line Interface (CLI)
<br><br>
The project includes an interactive CLI where users can ask questions and receive responses in real time.This interface allows for dynamic user interaction and makes it easy to query the system without additional UI development.
<br>
API Key Configuration
<br>
The Gemini API key should be stored securely using a .env file:
<br>
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
<br>
Optimize FAISS retrieval with Hierarchical Navigable Small World (HNSW) indexing.
<br>
Use better chunking strategies, such as semantic segmentation instead of fixed-size chunks.
<br>
Replace the Gemini API with open-source LLMs for cost efficiency.
<br>
Build a web-based UI using Streamlit for easier interaction.
<br><br>
<b>License</b>
<br><br>
This project is open-source and available under the MIT License.
<br><br>
<b>Acknowledgments</b>
<br><br>
https://python.langchain.com/docs/integrations/document_loaders/pdfplumber/ for PDF text extraction.
<br>
https://sbert.net/ for generating embeddings.
<br>
https://ai.meta.com/tools/faiss/ for efficient vector search.
<br>
https://ai.google.dev/ for text generation.
