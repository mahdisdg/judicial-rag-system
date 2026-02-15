import streamlit as st
import sys
import os
import time
import gc
from pathlib import Path
from dotenv import load_dotenv

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from config.config import Config
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

# ==============================================================================
# SAFE CACHING STRATEGY
# ==============================================================================

# LOAD RERANKER
@st.cache_resource(show_spinner="در حال بارگذاری Reranker...")
def get_reranker():
    return ReRanker("BAAI/bge-reranker-v2-m3")

# LOAD SEARCH ENGINE (Embedder + Qdrant Connection)
# We MUST cache the Retriever to prevent Qdrant from trying to open the DB twice.
@st.cache_resource(show_spinner="در حال اتصال به پایگاه داده و مدل Embedding...")
def get_search_engine(model_name):
    # Explicit garbage collection to release old locks/memory if switching
    gc.collect()
    
    print(f"🔄 INITIALIZING ENGINE FOR: {model_name}")
    
    # Determine Config
    is_e5 = "e5" in model_name.lower()
    clean_name = model_name.split("/")[-1]
    collection_name = f"legal_rag_{clean_name}"
    db_path = Config.DB_ROOT_DIR / f"qdrant_{clean_name}"

    if not db_path.exists():
        return None, None, f"پایگاه داده برای {clean_name} یافت نشد."

    # Load Model
    embedder = Embedder(model_name=model_name, is_e5=is_e5)
    
    # Connect to Qdrant
    retriever = Retriever(
        qdrant_path=str(db_path), 
        collection_name=collection_name, 
        embedder=embedder
    )
    
    return embedder, retriever, None

# LOAD LLM
def get_llm_client(model_name, temp, top_p, api_key):
    return LLMClient(
        model_name=model_name, 
        api_key=api_key,
        temperature=temp,
        top_p=top_p
    )

# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ تنظیمات سیستم")
    
    selected_llm = st.selectbox(
        "مدل زبانی (LLM)",
        ["gpt-4o-mini", "qwen2.5-vl-3b-instruct"],
        index=0
    )

    selected_embedding = st.selectbox(
        "مدل امبدینگ (Embedding)",
        [
            "HooshvareLab/bert-base-parsbert-uncased",
            "intfloat/multilingual-e5-base"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("پارامترهای تولید")
    temperature = st.slider("میزان خلاقیت (Temperature)", 0.0, 1.0, 0.0, 0.1)
    top_p = st.slider("تنوع پاسخ (Top P)", 0.0, 1.0, 0.9, 0.05)

    st.markdown("---")
    if st.button("پاک‌سازی حافظه گفتگو"):
        st.session_state.messages = []
        st.rerun()

# --- INIT ---
load_dotenv()
api_key = os.getenv("AVALAI_API_KEY")
if not api_key:
    st.error("❌ کلید API یافت نشد.")
    st.stop()

# Get Cached Resources
try:
    reranker = get_reranker()
    embedder, retriever, error_msg = get_search_engine(selected_embedding)
    
    if error_msg:
        st.error(f"⛔ خطا: {error_msg}")
        st.warning("لطفاً ابتدا اسکریپت indexing/run_indexing.py را برای این مدل اجرا کنید.")
        st.stop()

except Exception as e:
    st.error(f"خطای سیستمی: {e}")
    st.stop()

# Build Pipeline
retrieval_pipe = RetrievalPipeline(retriever, reranker, embedder)
llm = get_llm_client(selected_llm, temperature, top_p, api_key)
rag = RAGPipeline(retrieval_pipeline=retrieval_pipe, llm_client=llm)

# --- UI LAYOUT ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚖️ دستیار هوشمند قضایی</h1>
    <div class="sub-title">گفتگوی حقوقی با هوش مصنوعی</div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RENDER HELPER ---
def render_citations(result_data):
    if not result_data.get('documents'): return

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
            
            source_url = data['metadata'].get('source_url', '#')
            score = f"{data['score']:.4f}"

            css_class = "flash-card-container cited" if is_cited else "flash-card-container"
            badge = f'<div class="citation-badge">استناد شده</div>' if is_cited else ""
            
            html = f"""<a href="{source_url}" target="_blank" class="doc-link"><div class="{css_class}">{badge}<div class="card-title">{label} | {title}</div><div class="card-meta"><span>پرونده: {raw_id}</span><span class="card-score">{score}</span></div></div></a>"""

            with cols[i % 3]:
                st.markdown(html, unsafe_allow_html=True)

# --- CHAT LOOP ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "rag_result" in message:
            render_citations(message["rag_result"])

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
            
            chat_history_for_llm = [
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages[:-1]
            ]
            
            start_time = time.time()
            result = rag.run(query, chat_history_for_llm)
            end_time = time.time()
            
            status.write(f"🔍 جستجو برای: **{result['rewritten_query']}**")
            status.write("📚 در حال بازیابی اسناد مرتبط...")
            status.update(label="پاسخ آماده شد", state="complete", expanded=False)
            
            placeholder.markdown(result['answer'])
            render_citations(result)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "rag_result": result 
            })

        except Exception as e:
            status.update(label="خطا", state="error")
            placeholder.error(f"خطا: {e}")