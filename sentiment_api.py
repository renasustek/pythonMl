from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize FastAPI and Sentiment Analyzer
app = FastAPI()
sia = SentimentIntensityAnalyzer()

# Request Model
class SentimentRequest(BaseModel):
    message: str

# Response Model
class SentimentResponse(BaseModel):
    score: float  # Only returns the score now

@app.post("/analyse_sentiment")
async def analyze_sentiment(request: SentimentRequest):
    sentiment_score = sia.polarity_scores(request.message)["compound"]
    return SentimentResponse(score=sentiment_score)

# Run with: uvicorn sentiment_api:app --host 0.0.0.0 --port 5002 --reload
