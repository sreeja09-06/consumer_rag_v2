import os
import json
import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai

# ---- SETTINGS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
CHUNKS_JSON = os.path.join(BASE_DIR, "chunks.json")
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # used automatically once deployed on Streamlit Cloud
except Exception:
    GEMINI_API_KEY = "PASTE_YOUR_API_KEY_HERE"   # used for local testing - just paste your key here
GEMINI_MODEL = "gemini-3.5-flash-lite"

TOP_K_EACH = 10
FINAL_TOP_K = 4  # non-Act slots; one Act chunk is added on top, guaranteed
# -------------------


@st.cache_resource
def load_resources():
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client_db = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client_db.get_or_create_collection(name="consumer_law")
    client = genai.Client(api_key=GEMINI_API_KEY)
    return chunks, bm25, embed_model, collection, client


def call_gemini_with_retry(client, model, contents, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(20 * (attempt + 1))
            else:
                raise


def translate_to_english(client, text):
    prompt = f"""Translate the following text to English. If it is already in English, return it unchanged.
Only output the translation, nothing else.

TEXT: {text}"""
    response = call_gemini_with_retry(client, GEMINI_MODEL, prompt)
    return response.text.strip()


def bm25_search(query, bm25, chunks, top_k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]


def semantic_search(query, embed_model, collection, top_k):
    query_embedding = embed_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "id": int(results["ids"][0][i]),
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "source_type": results["metadatas"][0][i]["source_type"]
        })
    return chunks


def merge_results_with_guaranteed_act(bm25_results, semantic_results, final_top_k, embed_model, collection):
    """Same RRF merge as before, but guarantees at least one Act chunk
    is included, instead of relying only on a ranking boost."""
    scores = {}
    chunk_lookup = {}
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + 60)
        chunk_lookup[cid] = chunk
    for rank, chunk in enumerate(semantic_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + 60)
        chunk_lookup[cid] = chunk

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    merged = [chunk_lookup[cid] for cid in ranked_ids[:final_top_k]]

    has_act = any(c["source_type"] == "act" for c in merged)
    if not has_act:
        act_only = [c for c in chunk_lookup.values() if c["source_type"] == "act"]
        if act_only:
            best_act = act_only[0]
            merged.append(best_act)

    return merged


def build_chat_prompt(conversation_history, chunks):
    context_text = ""
    for i, chunk in enumerate(chunks, 1):
        context_text += f"\n[Excerpt {i} - from {chunk['source']}]\n{chunk['text']}\n"

    system_instructions = f"""You are a helpful assistant explaining Indian consumer rights and complaint procedures in plain language.

You are having a CONVERSATION with the user, not just answering one message in isolation.

BEHAVIOR:
- If the user's problem is missing key details (when they bought/used the product or service, where from, and exactly what went wrong), ask 1-2 short, specific clarifying questions FIRST. Do not give full guidance yet.
- - Once you have enough details (or the user says they don't know / want to skip), give a FINAL STRUCTURED ANSWER with exactly these four sections.
- FORMAT RULES for the final answer: write each section heading on its OWN line as a markdown heading, exactly as "## 1. Problem Summary", "## 2. Relevant Law", "## 3. Similar Cases" and "## 4. Recommended Steps". Leave a blank line after every heading and start that section's content on a new line below it. Never put any text, notes or citations on the same line as a heading. Put notes like "the excerpts do not mention a section number" inside the section's body as normal text, in their own paragraph. Use bullet points for lists of steps or points.
- Use ONLY the excerpts provided below for legal content. If they don't cover something, say so honestly.
- Some excerpts may reference the older Consumer Protection Act, 1986. If used, explicitly note it has since been replaced by the Consumer Protection Act, 2019.
- Do NOT turn a case-specific detail (like a specific number of days) into a general rule unless the excerpt states it as a general legal rule.
- Whenever an excerpt contains a specific section number, rule number, or clause (e.g. "Section 2(47)", "Rule 5", "Section 17(1)(a)"), CITE IT EXPLICITLY BY NAME in your answer, not just by excerpt number. If no specific section/rule number appears in the relevant excerpt, say so rather than inventing one.
- Respond in the SAME LANGUAGE the user is currently writing in.
- Mention which excerpt number(s) support your Relevant Law and Similar Cases sections, in addition to any section/rule numbers.

EXCERPTS AVAILABLE:
{context_text}
"""

    contents = [{"role": "user", "parts": [{"text": system_instructions}]}]
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    return contents


def get_retrieval_query(conversation_history):
    user_messages = [m["content"] for m in conversation_history if m["role"] == "user"]
    return " ".join(user_messages)


# ---- Streamlit UI ----
st.set_page_config(page_title="Consumer Court Chatbot", page_icon="⚖️")

# ---- Login gate: nothing below runs until the user signs in with Google ----
if not st.user.is_logged_in:
    st.title("⚖️ Consumer Court Chatbot")
    st.write("Sign in with Google to ask questions and keep your chat history.")
    st.button("Continue with Google", on_click=st.login)
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.user.name}**")
    st.caption(st.user.email)
    st.button("Log out", on_click=st.logout)

st.title("⚖️ AI Legal Assistant — Consumer Courts")
st.caption("Describe your issue. I may ask a couple of follow-up questions before giving full guidance.")

chunks, bm25, embed_model, collection, client = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Describe your consumer issue...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieval_query = get_retrieval_query(st.session_state.messages)
            translated_query = translate_to_english(client, retrieval_query)

            bm25_results = bm25_search(translated_query, bm25, chunks, TOP_K_EACH)
            semantic_results = semantic_search(translated_query, embed_model, collection, TOP_K_EACH)
            merged = merge_results_with_guaranteed_act(
                bm25_results, semantic_results, FINAL_TOP_K, embed_model, collection
            )

            contents = build_chat_prompt(st.session_state.messages, merged)
            response = call_gemini_with_retry(client, GEMINI_MODEL, contents)

            st.write(response.text)

            with st.expander("See retrieved sources"):
                for c in merged:
                    st.write(f"- **{c['source_type']}**: {c['source']}")

    st.session_state.messages.append({"role": "assistant", "content": response.text})

st.divider()
if st.button("Start New Conversation"):
    st.session_state.messages = []
    st.rerun()

st.caption("This assistant explains consumer rights and procedures; it does not give legal verdicts or predict case outcomes.")
