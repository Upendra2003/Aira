from flask import Flask, request, jsonify
import os
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

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")

if not MONGO_URI:
    raise ValueError("MongoDB connection string is missing. Check your .env file.")

app.config["MONGO_URI"] = MONGO_URI
mongo = PyMongo(app)
db = mongo.cx.get_database("AIRA")  # Explicitly get the database
chat_history_collection = db["chat_history"]
feedback_collection = db["feedback"]

# AI Model Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
output_parser = StrOutputParser()

# Initialize embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def get_retriever():
    vector_store = FAISS.load_local(
        "faiss_therapist_replies", embeddings=embedding_model, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

retriever = get_retriever()


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

# Format Retrieved Documents
def format_retrieved(docs):
    return " ".join([doc.page_content.replace("\n", " ") for doc in docs if hasattr(doc, "page_content")])

# Store chat history per user session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    session = chat_history_collection.find_one({"session_id": session_id})
    history = ChatMessageHistory()
    if session:
        for msg in session["messages"]:
            history.add_user_message(msg["message"]) if msg["role"] == "user" else history.add_ai_message(msg["message"])
    return history

# Runnable with Chat History
rag_chain = RunnableWithMessageHistory(
    RunnableMap({
        "context": lambda x: format_retrieved(retriever.invoke(x["input"])),
        "input": lambda x: x["input"],
        "chat_history": lambda x: [msg.content for msg in get_session_history(x["session_id"]).messages],
    })
    | prompt
    | model
    | output_parser,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Store chat history in MongoDB
def store_chat_history(session_id, user_input, ai_response):
    chat_history_collection.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"$each": [
            {"role": "user", "message": user_input},
            {"role": "AI", "message": ai_response}
        ]}}},
        upsert=True
    )

# Store feedback in MongoDB
def store_feedback(session_id, user_input, ai_response, feedback):
    feedback_collection.insert_one({
        "session_id": session_id,
        "user_input": user_input,
        "ai_response": ai_response,
        "feedback": feedback
    })

# Chat Route
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    chat_history = get_session_history(session_id)
    response = rag_chain.invoke(
        {"input": user_input, "session_id": session_id},
        config={"configurable": {"session_id": session_id}}
    )
    store_chat_history(session_id, user_input, response)

    return jsonify({
        "response": response,
        "chat_history": [{"role": "user", "message": msg.content} if msg.type == "human" else {"role": "AI", "message": msg.content} for msg in chat_history.messages]
    })

# Feedback Route
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.json
    session_id = data.get("session_id", "default")
    user_input = data.get("input", "")
    ai_response = data.get("response", "")
    feedback_type = data.get("feedback")  # "like" or "dislike"

    if not user_input or not ai_response or feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "Invalid feedback data"}), 400

    store_feedback(session_id, user_input, ai_response, feedback_type)
    return jsonify({"message": "Feedback recorded successfully"})

# Retrieve all feedback
@app.route("/api/feedback-summary", methods=["GET"])
def feedback_summary():
    feedback_data = list(feedback_collection.find({}, {"_id": 0}))
    return jsonify(feedback_data)

if __name__ == "__main__":
    app.run()
