import pandas as pd
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load cognitive restructuring dataset
csv_path = "Datasets/cog_restruct.csv"  # Ensure this file exists
df = pd.read_csv(csv_path)

# Ensure necessary columns exist
if "Thought" not in df.columns or "Reframe" not in df.columns:
    raise ValueError("Dataset must contain 'Thought' and 'Reframe' columns.")

# Preprocess text
def preprocess(text):
    return text.lower().strip()

df["Thought"] = df["Thought"].apply(preprocess)
df["Reframe"] = df["Reframe"].apply(preprocess)

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# Tokenize and encode text
thoughts = list(df["Thought"])
thought_encodings = tokenizer(thoughts, padding=True, truncation=True, return_tensors="tf")

# Generate BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
    outputs = bert_model(tokens.input_ids)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use CLS token embedding

# Compute embeddings for dataset
thought_embeddings = np.array([get_bert_embedding(thought) for thought in thoughts]).squeeze()

# Function to find the best response
def get_restructured_thought(user_input):
    user_embedding = get_bert_embedding(preprocess(user_input))
    similarities = cosine_similarity(user_embedding.reshape(1, -1), thought_embeddings)
    best_match_idx = np.argmax(similarities)
    return df.iloc[best_match_idx]["Reframe"]

# Simple chatbot loop
def chatbot():
    print("Welcome to the Cognitive Restructuring Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye! Remember to challenge your negative thoughts!")
            break
        response = get_restructured_thought(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chatbot()
