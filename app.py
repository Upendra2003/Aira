from flask import Flask, request, jsonify
import os
import time
import logging
import psutil
import gc
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableMap
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from flask_pymongo import PyMongo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not MONGO_URI:
    raise ValueError("MongoDB connection string is missing. Check your .env file.")

if not GROQ_API_KEY:
    raise ValueError("GROQ API key is missing. Check your .env file.")

app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)
db = mongo.cx.get_database("AIRA")
chat_history_collection = db["chat_history"]
feedback_collection = db["feedback"]

# Global variables with lazy loading
model = None
embedding_model = None
retriever = None
session_cache = {}

def get_model():
    """Lazy load the Groq LLM model."""
    global model
    if model is None:
        logger.info("Initializing Groq LLM model")
        model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
    return model

def get_embedding_model():
    """Lazy load the HuggingFace embedding model."""
    global embedding_model
    if embedding_model is None:
        logger.info("Initializing embedding model")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def get_retriever():
    """Lazy load the FAISS retriever."""
    global retriever
    if retriever is None:
        logger.info("Initializing FAISS retriever")
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(
            "faiss_therapist_replies", 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever

output_parser = StrOutputParser()

# System Prompt
system_prompt = """🌿 You are AIRA, an AI therapist dedicated to supporting individuals in their emotional well-being and mental health. Your role is to provide a safe, supportive, and judgment-free space for users to express their concerns. 🤗💙

📝 Guidelines:
✅ Maintain Context: Remember and reference relevant details from previous messages. 🧠💡
✅ Stay Engaged: Keep track of the conversation flow and respond accordingly. 🔄💬
✅ Be Clear & Concise: Use direct, to-the-point responses while maintaining warmth and empathy. ❤️✨
✅ Use Natural Language: Prioritize easy-to-understand language while ensuring depth and professionalism. 🗣️📖
✅ Encourage Professional Help When Necessary: If a user's concern requires medical attention, gently suggest seeking professional help. 🏥💙
✅ Use Emojis Thoughtfully: Incorporate emojis 😊🌸💖 when appropriate to build an emotional connection with the user and make the conversation feel more engaging and supportive.

🚧 Boundaries:
🚫 If users ask about unrelated topics (e.g., movies 🎬, anime 🎭, games 🎮, general queries 🌍, etc.) or anything outside of mental health, kindly inform them that you are designed solely for mental health support. 🧘‍♂️💙"""

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def format_retrieved(docs):
    """Format retrieved documents into a single string."""
    return " ".join([doc.page_content.replace("\n", " ") for doc in docs if hasattr(doc, "page_content")])

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get chat history for a session, with caching."""
    # Check if we have this session cached
    if session_id in session_cache:
        cache_time, history = session_cache[session_id]
        # If cache is less than 5 minutes old, use it
        if time.time() - cache_time < 300:
            return history
    
    # If not cached or cache expired, fetch from DB
    history = ChatMessageHistory()
    try:
        session = chat_history_collection.find_one({"session_id": session_id})
        if session:
            for msg in session.get("messages", []):
                if msg["role"] == "user":
                    history.add_user_message(msg["message"])
                else:
                    history.add_ai_message(msg["message"])
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
    
    # Update cache
    session_cache[session_id] = (time.time(), history)
    
    # Clean up old cache entries
    clean_session_cache()
    
    return history

def clean_session_cache():
    """Remove old sessions from cache to prevent memory growth."""
    current_time = time.time()
    # Remove sessions older than 10 minutes
    expired_sessions = [
        sid for sid, (timestamp, _) in session_cache.items() 
        if current_time - timestamp > 600
    ]
    for sid in expired_sessions:
        del session_cache[sid]

def create_chain():
    """Create the LangChain chain on demand."""
    return RunnableWithMessageHistory(
        RunnableMap({
            "context": lambda x: format_retrieved(get_retriever().invoke(x["input"])),
            "input": lambda x: x["input"],
            "chat_history": lambda x: [msg.content for msg in get_session_history(x["session_id"]).messages],
        })
        | prompt
        | get_model()
        | output_parser,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

def store_chat_history(session_id, user_input, ai_response):
    """Store chat history in MongoDB."""
    try:
        chat_history_collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"$each": [
                {"role": "user", "message": user_input},
                {"role": "AI", "message": ai_response}
            ]}}},
            upsert=True
        )
        
        # Update cache if it exists
        if session_id in session_cache:
            _, history = session_cache[session_id]
            history.add_user_message(user_input)
            history.add_ai_message(ai_response)
            session_cache[session_id] = (time.time(), history)
    except Exception as e:
        logger.error(f"Error storing chat history: {e}")

def store_feedback(session_id, user_input, ai_response, feedback):
    """Store feedback in MongoDB."""
    try:
        feedback_collection.insert_one({
            "session_id": session_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "feedback": feedback,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

# Routes
@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests."""
    start_time = time.time()
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    logger.info(f"Chat request for session {session_id}")
    
    try:
        # Create chain on demand
        rag_chain = create_chain()
        
        # Get chat history
        chat_history = get_session_history(session_id)
        
        # Generate response
        response = rag_chain.invoke(
            {"input": user_input, "session_id": session_id},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Store in database
        store_chat_history(session_id, user_input, response)
        
        # Force garbage collection to free memory
        gc.collect()
        
        process_time = time.time() - start_time
        logger.info(f"Chat processed in {process_time:.2f} seconds")
        
        return jsonify({
            "response": response,
            "chat_history": [
                {"role": "user", "message": msg.content} if msg.type == "human" 
                else {"role": "AI", "message": msg.content} 
                for msg in chat_history.messages
            ]
        })
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Handle feedback submissions."""
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")
    ai_response = data.get("response", "")
    feedback_type = data.get("feedback")  # "like" or "dislike"

    if not user_input or not ai_response or feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "Invalid feedback data"}), 400

    store_feedback(session_id, user_input, ai_response, feedback_type)
    return jsonify({"message": "Feedback recorded successfully"})

@app.route("/api/feedback-summary", methods=["GET"])
def feedback_summary():
    """Get feedback summary with pagination."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        
        skip = (page - 1) * per_page
        feedback_data = list(feedback_collection.find({}, {"_id": 0}).skip(skip).limit(per_page))
        
        count = feedback_collection.count_documents({})
        
        return jsonify({
            "feedback": feedback_data,
            "total": count,
            "page": page,
            "per_page": per_page,
            "pages": (count + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error getting feedback summary: {e}")
        return jsonify({"error": "An error occurred retrieving feedback"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "ok", 
        "timestamp": time.time(),
        "uptime": time.time() - app.start_time
    })

@app.route("/memory", methods=["GET"])
def memory_usage():
    """Memory usage monitoring endpoint."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return jsonify({
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "sessions_cached": len(session_cache)
    })

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Clean up resources when app context ends."""
    global model, embedding_model, retriever
    model = None
    embedding_model = None
    retriever = None
    gc.collect()

if __name__ == "__main__":
    # Set app start time for uptime monitoring
    app.start_time = time.time()
    logger.info("Starting AIRA Therapist application")
    
    # Get port from environment variable (for Render compatibility)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)