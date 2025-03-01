from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initialize FastAPI app
app = FastAPI()

# Load cognitive restructuring dataset
csv_path = "Datasets/cog_restruct.csv"  # Ensure this file exists
df = pd.read_csv(csv_path)

# Ensure necessary columns exist
if "Thought" not in df.columns or "Reframe" not in df.columns:
    raise ValueError("Dataset must contain 'Thought' and 'Reframe' columns.")

# Preprocess text
def preprocess(text):
    return str(text).lower().strip()  # Ensure text is string to avoid NoneType errors

df["Thought"] = df["Thought"].apply(preprocess)
df["Reframe"] = df["Reframe"].apply(preprocess)

# Convert thoughts into vector space
vectorizer = TfidfVectorizer()
thought_vectors = vectorizer.fit_transform(df["Thought"])

# Request model
class ChatRequest(BaseModel):
    message: str

# Function to find the best response
def get_restructured_thought(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, thought_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx]["Reframe"]

# FastAPI route for chatbot response
@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    response = get_restructured_thought(request.message)
    return {"response": response}

# Run using: uvicorn simple_chatbot:app --host 0.0.0.0 --port 5001 --reload
