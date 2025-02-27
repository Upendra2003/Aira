# AIRA Therapist - AI Mental Health Support

AIRA is an AI therapist application designed to provide mental health support through a conversational interface. It uses the Groq LLM API for natural language processing, FAISS for vector similarity search, and MongoDB for data storage.

## Features

- Conversational AI therapist using Groq's Llama3-8b model
- Memory of conversation history for contextual responses
- RAG (Retrieval Augmented Generation) with FAISS vector database
- User feedback collection system

## Requirements

- Python 3.9+
- MongoDB database (can use MongoDB Atlas free tier)
- Groq API key

## Setup Instructions

1. Clone the repository
2. Create a `.env` file with the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   MONGO_CONNECTION_STRING=your_mongodb_connection_string
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run locally:
   ```
   python app.py
   ```

## Deployment to Render

1. Connect your GitHub repository to Render
2. Set the required environment variables in the Render dashboard
3. Deploy using the settings in `render.yaml`

## API Endpoints

- `POST /api/chat` - Send a message to AIRA
  ```json
  {
    "session_id": "unique_user_id",
    "input": "User message here"
  }
  ```

- `POST /api/feedback` - Submit feedback for a response
  ```json
  {
    "session_id": "unique_user_id",
    "input": "Original user message",
    "response": "AI response",
    "feedback": "like" or "dislike"
  }
  ```

- `GET /api/feedback-summary?page=1&per_page=50` - Get feedback summary with pagination

- `GET /health` - Health check endpoint

- `GET /memory` - Memory usage monitoring

## Memory Optimization

This application is optimized for Render's free tier with the following strategies:

1. Lazy loading of models and components
2. Session caching with expiration
3. Garbage collection after requests
4. Single worker with thread-based concurrency
5. Memory monitoring endpoints


