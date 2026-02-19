"""
retriever.py
------------
Loads the FAISS vector store, retrieves relevant document chunks
for a user query, and generates a response using a local Ollama LLM.
"""

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from src.memory.short_term_memory import fetch_conversation_history, format_history_for_prompt

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL       = "phi3:mini"
VECTOR_STORE_PATH  = Path(__file__).parent.parent.parent / "vector_store"
RELEVANCE_THRESHOLD = 0.4

# ── Load resources once at module level ──────────────────────────────────────
logger.info("🔄 Loading embedding model and FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db  = FAISS.load_local(
    str(VECTOR_STORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)
llm = Ollama(model=OLLAMA_MODEL)
logger.info("✅ Resources loaded successfully.")

# ── Prompt Template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=(
        "You are a concise IT support assistant. "
        "Use the provided context to answer the user's question in clear, stepwise format. "
        "Use relevant emojis where appropriate.\n\n"
        "If the question is unrelated to the context, respond ONLY with:\n"
        "'Sorry, I don't have knowledge about this question.'\n\n"
        "Conversation History:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Stepwise Answer:"
    )
)


def retrieve_relevant_context(query: str) -> list:
    """
    Search the FAISS index and return filtered relevant document chunks.
    Uses a dynamic threshold based on the top similarity score.
    """
    docs_and_scores = vector_db.similarity_search_with_score(query, k=3)

    if not docs_and_scores:
        logger.warning("⚠️ No documents returned from vector search.")
        return []

    top_score = max(score for _, score in docs_and_scores)
    dynamic_threshold = max(RELEVANCE_THRESHOLD, top_score * 0.7)
    logger.info(f"📊 Top score: {top_score:.4f} | Dynamic threshold: {dynamic_threshold:.4f}")

    relevant_docs = [doc for doc, score in docs_and_scores if score >= dynamic_threshold]
    logger.info(f"📄 Relevant chunks retained: {len(relevant_docs)}")
    return relevant_docs


def generate_answer(query: str, sender_id: str) -> str:
    """
    Full RAG pipeline:
      1. Fetch short-term conversation memory.
      2. Retrieve relevant context from FAISS.
      3. Generate answer via Ollama LLM.

    Args:
        query:     The user's question.
        sender_id: Unique session/user identifier for memory lookup.

    Returns:
        A string answer from the LLM, or an error message.
    """
    logger.info(f"🔍 Processing query for sender '{sender_id}': {query}")

    # Step 1: Short-term memory
    history_messages  = fetch_conversation_history(sender_id)
    formatted_history = format_history_for_prompt(history_messages)

    # Step 2: Context retrieval
    relevant_docs = retrieve_relevant_context(query)
    if not relevant_docs:
        return "❌ Sorry, your question seems unrelated to my knowledge base."

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 3: LLM generation
    try:
        chain    = PROMPT_TEMPLATE | llm
        response = chain.invoke({
            "history":  formatted_history,
            "context":  context,
            "question": query
        })

        # Normalise LangChain response types
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("content") or str(response)
        return str(response)

    except Exception as e:
        logger.error(f"❌ LLM generation failed: {e}")
        return "❌ An error occurred while generating the response."
