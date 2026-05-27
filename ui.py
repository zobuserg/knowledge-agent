"""
Knowledge Agent — UI
Run: streamlit run ui.py
"""

import sys
import tempfile
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.pipeline import (
    sync_folder, ingest_files, get_index_stats, get_connected_folder
)
from app.query.agent import query, _engine_cache

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Knowledge Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #888; font-size: 0.95rem; margin-top: 0; }
    .stat-box { background: #1e1e2e; border-radius: 8px; padding: 12px 16px; margin: 4px 0; }
    .stat-number { font-size: 1.4rem; font-weight: 700; color: #cba6f7; }
    .stat-label { font-size: 0.75rem; color: #888; }
    .setup-card { background: #1e1e2e; border: 1px solid #313244;
                  border-radius: 12px; padding: 24px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ── State ─────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "last_stats" not in st.session_state:
    st.session_state.last_stats = None


def refresh_stats():
    st.session_state.last_stats = get_index_stats()
    return st.session_state.last_stats


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Knowledge Agent")

    stats = refresh_stats()
    has_docs = stats["total_chunks"] > 0

    # Stats
    if has_docs:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats['total_documents']}</div>
                <div class="stat-label">documents</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats['total_chunks']}</div>
                <div class="stat-label">chunks</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Tab: Folder sync ──────────────────────────────────────────────────────
    tab_folder, tab_files = st.tabs(["📁 Folder", "📄 Files"])

    with tab_folder:
        connected = stats.get("connected_folder")
        if connected:
            st.caption(f"Connected: `{Path(connected).name}`")
            st.caption(f"`{connected}`")
            if st.button("🔄 Sync now", use_container_width=True, type="primary"):
                with st.spinner("Scanning for new files..."):
                    result = sync_folder(connected)
                    _engine_cache.clear()
                if result["new"] == 0:
                    st.success(f"Already up to date ({result['skipped']} files unchanged)")
                else:
                    st.success(f"✅ {result['new']} new files indexed")
                st.rerun()
            st.divider()

        folder_path = st.text_input(
            "Connect a folder",
            placeholder="C:/Users/you/Documents  or  D:/vault",
            help="Paste the path to any folder on your computer",
        )
        recursive = st.checkbox("Include subfolders", value=True)

        if st.button("Connect & sync", use_container_width=True,
                     disabled=not folder_path.strip()):
            with st.spinner("Indexing folder..."):
                try:
                    result = sync_folder(folder_path.strip(), recursive=recursive)
                    _engine_cache.clear()
                    st.success(
                        f"✅ Done — {result['new']} files indexed"
                        + (f", {result['skipped']} unchanged" if result['skipped'] else "")
                    )
                    st.rerun()
                except FileNotFoundError:
                    st.error("Folder not found. Check the path and try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_files:
        uploaded = st.file_uploader(
            "Drop files here",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            help="PDF, Markdown, or plain text. Select multiple with Ctrl+click.",
        )
        if uploaded and st.button("📥 Add to knowledge base",
                                   use_container_width=True, type="primary"):
            with st.spinner(f"Indexing {len(uploaded)} file(s)..."):
                saved = []
                for f in uploaded:
                    suffix = Path(f.name).suffix
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix,
                        dir="data/raw", prefix=Path(f.name).stem + "_"
                    )
                    tmp.write(f.read())
                    tmp.close()
                    saved.append(Path(tmp.name))

                result = ingest_files(saved)
                _engine_cache.clear()
                st.success(f"✅ {result['new']} file(s) indexed")
                st.rerun()

    # ── Indexed documents list ────────────────────────────────────────────────
    if has_docs and stats.get("documents"):
        st.divider()
        with st.expander(f"📋 {stats['total_documents']} indexed documents", expanded=False):
            for doc in stats["documents"][:50]:
                st.caption(f"· {doc}")
            if stats['total_documents'] > 50:
                st.caption(f"... and {stats['total_documents'] - 50} more")

    st.divider()
    if st.button("🗑️ Clear session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = "default"
        _engine_cache.clear()
        st.rerun()

    st.caption("Runs 100% locally · Powered by Ollama")


# ── Main: onboarding or chat ──────────────────────────────────────────────────

if not has_docs:
    # First-run onboarding
    st.markdown('<p class="main-header">🧠 Knowledge Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents. Answers with citations.</p>',
                unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="setup-card">
            <h3>📁 Connect a folder</h3>
            <p>Have an existing knowledge base? Obsidian vault, documents folder, project notes?</p>
            <p>Paste the folder path in the sidebar → <b>Connect & sync</b></p>
            <p style="color:#888; font-size:0.85rem;">All files stay where they are. Nothing gets moved.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="setup-card">
            <h3>📄 Upload files</h3>
            <p>Don't have a folder yet? Drop individual files directly — PDF, Markdown, or plain text.</p>
            <p>Go to <b>Files tab</b> in the sidebar and drag your documents.</p>
            <p style="color:#888; font-size:0.85rem;">Select multiple files with Ctrl+click.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Once connected, you can ask things like:")
    examples = [
        "What did I learn about LangGraph?",
        "Summarize the main points of [document name]",
        "What are my notes on RAG architecture?",
        "Find everything related to Python async",
    ]
    for ex in examples:
        st.markdown(f"· *{ex}*")

else:
    # Chat interface
    st.markdown('<p class="main-header">🧠 Knowledge Agent</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-header">{stats["total_documents"]} documents · {stats["total_chunks"]} chunks indexed</p>',
        unsafe_allow_html=True
    )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in msg["sources"]:
                        score_str = f"  `{s['score']}`" if s.get("score") else ""
                        st.markdown(f"**{s['file']}**{score_str}")
                        if s.get("excerpt"):
                            st.caption(s["excerpt"])

    # Input
    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    result = query(question, session_id=st.session_state.session_id)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for s in sources:
                                score_str = f"  `{s['score']}`" if s.get("score") else ""
                                st.markdown(f"**{s['file']}**{score_str}")
                                if s.get("excerpt"):
                                    st.caption(s["excerpt"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    msg = f"Error: {e}"
                    st.error(msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": msg
                    })
