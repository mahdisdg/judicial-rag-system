import streamlit as st
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from indexing.src.config import Config
from indexing.src.embedding import Embedder
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker
from retrieval.src.pipeline import RetrievalPipeline
from rag_llm.src.llm_client import LLMClient
from rag_llm.src.rag_pipeline import RAGPipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="دستیار قضایی", page_icon="⚖️", layout="wide")

# --- CSS LOADING ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent / "assets" / "style.css"
local_css(css_path)

# --- CACHED PIPELINE ---
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    load_dotenv()
    api_key = os.getenv("AVALAI_API_KEY")
    if not api_key: return None
    
    print("⏳ Loading Models...")
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    retriever = Retriever(str(Config.QDRANT_PATH), Config.COLLECTION_NAME, embedder)
    reranker = ReRanker("BAAI/bge-reranker-v2-m3")
    pipe = RetrievalPipeline(retriever, reranker, embedder)
    llm = LLMClient(model_name="gpt-4o-mini", api_key=api_key)
    return RAGPipeline(pipe, llm)

# --- INIT ---
if "rag" not in st.session_state:
    with st.spinner("در حال راه‌اندازی مغز حقوقی سیستم..."):
        st.session_state.rag = get_rag_pipeline()

rag = st.session_state.rag

if not rag:
    st.error("❌ کلید API یافت نشد.")
    st.stop()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RENDER FUNCTION ---
def render_citations(result_data):
    if not result_data.get('documents'):
        return

    st.markdown("---")
    st.markdown("##### 📚 مستندات و منابع")
    
    used = result_data['used_docs']
    all_docs = list(result_data['documents'].keys())
    sorted_docs = sorted(all_docs, key=lambda x: 0 if x in used else 1)
    
    with st.container():
        cols = st.columns(3)
        for i, label in enumerate(sorted_docs):
            data = result_data['documents'][label]
            is_cited = label in used
            
            title = data['metadata'].get('title', 'بدون عنوان')
            raw_id = data['real_doc_id']
            doc_id = raw_id
            
            # Source URL
            source_url = data['metadata'].get('source_url', '#')
            
            score = f"{data['score']:.4f}"

            css_class = "flash-card-container cited" if is_cited else "flash-card-container"
            badge = f'<div class="citation-badge">استناد شده</div>' if is_cited else ""
            
            html = f"""<a href="{source_url}" target="_blank" class="doc-link"><div class="{css_class}">{badge}<div class="card-title">{label} | {title}</div><div class="card-meta"><span>پرونده: {doc_id}</span><span class="card-score">{score}</span></div></div></a>"""

            
            with cols[i % 3]:
                st.markdown(html, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ تنظیمات")
    st.selectbox("مدل زبانی", ["GPT-4o-mini"], disabled=True)
    st.selectbox("روش جستجو", ["Hybrid Search"], disabled=True)
    st.divider()
    if st.button("پاک‌سازی حافظه"):
        st.session_state.messages = []
        st.rerun()

# --- UI LAYOUT ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚖️ دستیار هوشمند قضایی</h1>
    <div class="sub-title">گفتگوی حقوقی با هوش مصنوعی</div>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "rag_result" in message:
            render_citations(message["rag_result"])

# Chat Input
query = st.chat_input("سوال خود را بپرسید...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("در حال تحلیل...", expanded=True)
        
        try:
            status.write("🧠 در حال درک منظور شما...")
            
            # Prepare history (excluding current query)
            chat_history_for_llm = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages[:-1]
            ]
            
            # Run Pipeline
            start_time = time.time()
            result = rag.run(query, chat_history_for_llm)
            end_time = time.time()
            
            status.write(f"🔍 جستجو برای: **{result['rewritten_query']}**")
            status.write("📚 در حال بازیابی اسناد مرتبط...")
            status.update(label="پاسخ آماده شد", state="complete", expanded=False)
            
            # Show Answer
            placeholder.markdown(result['answer'])
            
            # Show Citations
            render_citations(result)
            
            # Save to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "rag_result": result 
            })

        except Exception as e:
            status.update(label="خطا", state="error")
            placeholder.error(f"خطا: {e}")