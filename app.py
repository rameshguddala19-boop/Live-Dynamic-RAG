import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import advanced_rag as rag
import config
import prepare_data


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Live & Dynamic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    /* Cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
        margin-bottom: 1rem;
    }

    .answer-card {
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .strategy-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 0.5rem;
    }

    .small-label {
        font-size: 0.8rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .big-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📄 Document")

    uploaded_file = st.file_uploader(
        "Upload Employee PDF",
        type=["pdf"],
        help="Upload the employee PDF to process."
    )

    st.caption("PDF only • Maximum 200 MB")

    if uploaded_file is not None:

        st.write(
            f"**Selected:** {uploaded_file.name}"
        )

        if st.button(
            "Process PDF",
            type="primary",
            use_container_width=True
        ):

            save_path = (
                DATA_DIR / uploaded_file.name
            )

            try:

                with open(
                    save_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                with st.spinner(
                    "Processing document..."
                ):

                    row_count = (
                        prepare_data.ingest_pdf(
                            str(save_path)
                        )
                    )

                st.session_state.ingested_file = (
                    uploaded_file.name
                )

                st.session_state.last_result = None

                st.success(
                    f"Processed successfully.\n"
                    f"{row_count} employee records loaded."
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {e}"
                )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Live & Dynamic RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent document question answering with SQL,
    hybrid retrieval, and live data sources.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATASET STATUS
# =========================================================

st.markdown("### 📊 Dataset Status")

db_path = Path(
    config.DB_PATH
)

if db_path.exists():

    try:

        con = sqlite3.connect(
            config.DB_PATH
        )

        employee_count = con.execute(
            "SELECT COUNT(*) FROM employees"
        ).fetchone()[0]

        con.close()

    except Exception:
        employee_count = 0

else:

    employee_count = 0


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Employees",
        employee_count
    )

with col2:

    st.metric(
        "Database",
        "Ready" if db_path.exists() else "Not Ready"
    )

with col3:

    current_file = (
        st.session_state.ingested_file
        or "No document"
    )

    st.metric(
        "Document",
        "Loaded" if st.session_state.ingested_file else "Not Loaded"
    )

if st.session_state.ingested_file:

    st.caption(
        f"Current document: **{st.session_state.ingested_file}**"
    )


# =========================================================
# QUESTION SECTION
# =========================================================

st.markdown("---")

st.markdown("### 💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Type your question here...",
    label_visibility="collapsed"
)

ask_button = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True
)


# =========================================================
# ASK QUESTION
# =========================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not db_path.exists():

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        try:

            with st.spinner(
                "Analyzing your question..."
            ):

                result = rag.query_document(
                    question
                )

            st.session_state.last_result = result

        except Exception as e:

            st.error(
                f"Question processing failed: {e}"
            )


# =========================================================
# RESULT
# =========================================================

result = st.session_state.last_result

if result is not None:

    st.markdown("---")

    st.markdown("### ✅ Answer")

    answer = result.get(
        "answer",
        "No answer available."
    )

    st.markdown(
        f"""
        <div class="answer-card">
            <div style="font-size:1.05rem;">
                {answer}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Strategy + Count
    # -----------------------------------------------------

    strategy = result.get(
        "strategy",
        ""
    )

    record_count = result.get(
        "record_count"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="strategy-card">
                <div class="small-label">
                    Retrieval Strategy
                </div>
            """,
            unsafe_allow_html=True
        )

        strategy_name = {
            "structured_filter":
                "Structured Filter (SQL)",

            "aggregation":
                "Aggregation (Python Count)",

            "exact_lookup":
                "Exact Lookup",

            "hybrid_retrieval":
                "Hybrid Retrieval",

            "live_lookup":
                "Live Data Lookup"
        }.get(
            strategy,
            strategy
        )

        st.markdown(
            f"### {strategy_name}"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown(
            """
            <div class="strategy-card">
                <div class="small-label">
                    Matching Records
                </div>
            """,
            unsafe_allow_html=True
        )

        if record_count is None:
            value = "—"
        else:
            value = str(record_count)

        st.markdown(
            f'<div class="big-value">{value}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # Router filters
    # -----------------------------------------------------

    filters = result.get(
        "filters",
        []
    )

    if filters:

        with st.expander(
            "🔎 View Applied Filters"
        ):

            st.json(filters)


    # -----------------------------------------------------
    # Live data
    # -----------------------------------------------------

    live_evidence = result.get(
        "live_evidence"
    )

    if live_evidence:

        st.markdown("### 🌦️ Live Data")

        live_cols = st.columns(4)

        city = live_evidence.get(
            "city",
            "—"
        )

        temperature = live_evidence.get(
            "temp_c",
            "—"
        )

        humidity = live_evidence.get(
            "humidity",
            "—"
        )

        condition = live_evidence.get(
            "condition",
            "—"
        )

        with live_cols[0]:
            st.metric(
                "City",
                city
            )

        with live_cols[1]:
            st.metric(
                "Temperature",
                f"{temperature} °C"
                if temperature != "—"
                else "—"
            )

        with live_cols[2]:
            st.metric(
                "Humidity",
                f"{humidity}%"
                if humidity != "—"
                else "—"
            )

        with live_cols[3]:
            st.metric(
                "Condition",
                str(condition).title()
            )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources = result.get(
        "sources",
        []
    )

    if sources:

        with st.expander(
            "📚 Sources / Matching Records"
        ):

            try:

                if isinstance(
                    sources,
                    list
                ):

                    df = pd.DataFrame(
                        sources
                    )

                    if not df.empty:

                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.write(
                            "No source records."
                        )

                else:

                    st.write(
                        sources
                    )

            except Exception:

                st.write(
                    sources
                )


# =========================================================
# EMPTY STATE
# =========================================================

if result is None:

    st.markdown("---")

    st.markdown(
        """
        <div class="info-card">
            <div class="small-label">
                Ready
            </div>
            <div style="font-size:1rem; margin-top:0.4rem;">
                Upload a PDF, process it, and ask your question.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )