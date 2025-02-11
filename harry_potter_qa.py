import pdfplumber
def parse_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text=""
        for page in pdf.pages:
            text+=page.extract_text()
    return text

book_text=parse_pdf("Harry Potter and the Prisoner of Azkaban.pdf")

#data chunking and preprocessing
#objective:to split the data into smaller ,coherent chunks which will make efficient retrieval and easesemantic clarity during vectorization.

def chunk_text(text, chunk_size=150, overlap=20):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),chunk_size-overlap):
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks=chunk_text(book_text)

#convert the chunks into dense vector representations
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks, convert_to_tensor=True)

import faiss
import numpy as np
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv

embeddings_dim=embeddings.shape[1]
index=faiss.IndexFlatL2(embeddings_dim)
index.add(np.array(embeddings))

metadata = [{"text": chunk} for chunk in chunks]

def query_gemini(question: str):
    API_KEY = "AIzaSyCNiaxqC5A4FDqKBUeP7BnvwvohtsuuolE"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
    data = {
        "contents": [{
            "parts": [{"text": question}]
        }]
    }

    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        try:
            # Extract the text from the response structure
            answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return answer
        except (KeyError, IndexError) as e:
            return f"Error: Unexpected response structure - {e}. Response: {response.json()}"
    else:
        return f"API Error: {response.status_code} - {response.text}"



def query_rag(question: str):
    question_embedding = model.encode(question, convert_to_tensor=False)
    distances, indices = index.search(np.array([question_embedding]), k=5)
    retrieved_chunks = [metadata[i]["text"] for i in indices[0]]
    combined_chunks = "\n".join(retrieved_chunks)
    
    # Send the combined context to the Gemini API
    return query_gemini(combined_chunks)

if __name__ == "__main__":
    print("Welcome to the Harry Potter Q&A Assistant!")
    print("Type your question about the book, or type 'exit' to quit.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == "exit":
            print("Goodbye!")
            break
        response = query_rag(user_question)
        print(f"\nAnswer: {response}")
