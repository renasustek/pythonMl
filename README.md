Project Overview

This repository contains Python scripts for natural language processing tasks focused on supporting students. It includes a sentiment analysis API for calculating user mood and tracking procrastination levels, and a cognitive restructuring chatbot based on CBT principles to help reduce academic procrastination.

This project utilizes the following key technologies:

* **VADER Lexicon**: This is a lexicon and rule-based sentiment analysis tool. In this project, it is used by `sentiment_api.py` to calculate sentiment scores for user messages, which can then be used to track user mood and procrastination levels in another application.
* **TF-IDF (Term Frequency-Inverse Document Frequency)**: TF-IDF is employed to convert text (specifically, "Thought" entries from the `cog_restruct.csv` dataset) into numerical vector representations, enabling the calculation of cosine similarity for finding the best matching reframed thoughts, used for the chatbot
* **FastAPI**: This is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. Both `sentiment_api.py` and `simple_chatbot.py` leverage FastAPI to create their respective web endpoints, allowing them to serve as accessible services for sentiment analysis and chatbot interactions.
