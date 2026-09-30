"""
Advanced RAG core:
Router + SQL filters + exact lookup + hybrid retrieval + live lookup.

Public entry point:
    query_document(question) -> dict
"""

import json
import logging
import re
import sqlite3

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

import config
import live_data


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)

log = logging.getLogger("rag")

client = OpenAI()


# =========================================================
# VALID STRATEGIES
# =========================================================

VALID_STRATEGIES = [
    "structured_filter",
    "aggregation",
    "exact_lookup",
    "hybrid_retrieval",
    "live_lookup",
]


# =========================================================
# ROUTER PROMPT
# =========================================================

ROUTER_SYSTEM = """
You route questions between an employee database and an active
uploaded PDF document.

The application has two main knowledge sources:

1. Employee database
2. Active uploaded PDF

IMPORTANT:
If the question is about information that may come from the
uploaded document, use hybrid_retrieval.

Do NOT use exact_lookup merely because the question contains
an employee ID, employee name, or person name.

Use exact_lookup only when the question clearly asks for an
attribute of a person that belongs to the employee database.

Strategies:

1. structured_filter
   Use for employee database list/show questions with
   structured conditions.

   Example:
   "List Engineering employees on leave"

2. aggregation
   Use for employee database COUNT / AVERAGE / SUM / MIN / MAX.

   Example:
   "How many employees are on leave?"

3. exact_lookup
   Use only for one specific person's employee-database
   attribute.

   Example:
   "What is Sneha Malhotra's salary?"

4. hybrid_retrieval
   Use for:
   - uploaded PDF questions
   - document questions
   - QA/testing concepts
   - open-ended questions
   - comparative questions
   - narrative questions
   - questions whose answer may exist in the active document

5. live_lookup
   Use for real-time external data.

   Example:
   "What is the weather in Hyderabad right now?"

For live_lookup:
identify the city from the question.

For all other strategies:
city must be empty.

---------------------------------------------------------
FILTER GROUPS
---------------------------------------------------------

Express conditions as FILTER GROUPS.

Each group:

{
    "op": "and" or "or",
    "conditions": [...]
}

Each condition:

{
    "field": "...",
    "match": "eq" or "contains",
    "value": "..."
}

The GROUPS themselves are always combined with AND.

---------------------------------------------------------
STATUS RULES
---------------------------------------------------------

For employee status:

"On Leave"

field = "status"
value = "On Leave"

"Active"

field = "status"
value = "Active"

Example:

"How many employees are On Leave?"

must become:

[
    {
        "op": "and",
        "conditions": [
            {
                "field": "status",
                "match": "eq",
                "value": "On Leave"
            }
        ]
    }
]

---------------------------------------------------------
DEPARTMENT RULES
---------------------------------------------------------

"Engineering employees"

must become:

[
    {
        "op": "and",
        "conditions": [
            {
                "field": "department",
                "match": "eq",
                "value": "Engineering"
            }
        ]
    }
]

"Finance or IT"

must become:

[
    {
        "op": "or",
        "conditions": [
            {
                "field": "department",
                "match": "eq",
                "value": "Finance"
            },
            {
                "field": "department",
                "match": "eq",
                "value": "IT"
            }
        ]
    }
]

---------------------------------------------------------
COMPOUND RULE
---------------------------------------------------------

"(Finance OR IT) AND Bengaluru"

must become two groups:

[
    {
        "op": "or",
        "conditions": [
            {
                "field": "department",
                "match": "eq",
                "value": "Finance"
            },
            {
                "field": "department",
                "match": "eq",
                "value": "IT"
            }
        ]
    },
    {
        "op": "and",
        "conditions": [
            {
                "field": "location",
                "match": "eq",
                "value": "Bengaluru"
            }
        ]
    }
]

---------------------------------------------------------
EXACT LOOKUP RULE
---------------------------------------------------------

For exact_lookup:
identify the person's name.

Example:

"What is Sneha Malhotra's salary?"

should identify:

name = "Sneha Malhotra"

---------------------------------------------------------
AGGREGATION RULE
---------------------------------------------------------

For COUNT questions:

- conditions must go inside groups
- target_field must be empty

target_field is used only for:

AVERAGE
SUM
MIN
MAX

For COUNT:
target_field must always be empty.

---------------------------------------------------------
OUTPUT
---------------------------------------------------------

Return STRICT JSON:

{
    "strategy": "...",
    "groups": [],
    "city": "",
    "target_field": ""
}
"""


# =========================================================
# CLASSIFY QUERY
# =========================================================

def classify_query(question: str) -> dict:
    """
    Ask the LLM to classify the question.
    """

    try:
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": ROUTER_SYSTEM
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        data = json.loads(
            response.choices[0].message.content
        )

    except Exception as e:

        log.warning(
            "Router failed (%s); using hybrid_retrieval",
            e
        )

        return {
            "strategy": "hybrid_retrieval",
            "groups": [],
            "city": "",
            "target_field": ""
        }

    strategy = data.get(
        "strategy",
        ""
    )

    if strategy not in VALID_STRATEGIES:
        strategy = "hybrid_retrieval"

    groups = data.get(
        "groups",
        []
    ) or []

    city = data.get(
        "city",
        ""
    ) or ""

    target_field = data.get(
        "target_field",
        ""
    ) or ""

    # -----------------------------------------------------
    # COUNT SAFETY
    # -----------------------------------------------------

    q = question.lower()

    count_words = [
        "how many",
        "number of",
        "count",
        "total number"
    ]

    if any(
        word in q
        for word in count_words
    ):
        target_field = ""

    return {
        "strategy": strategy,
        "groups": groups,
        "city": city,
        "target_field": target_field
    }


# =========================================================
# DATABASE COLUMNS
# =========================================================

def _column_names() -> list[str]:
    """
    Get employee table columns.
    """

    con = sqlite3.connect(
        config.DB_PATH
    )

    columns = [
        row[1]
        for row in con.execute(
            "PRAGMA table_info(employees)"
        )
    ]

    con.close()

    return columns


# =========================================================
# APPLY FILTERS
# =========================================================

def apply_filters(
    groups: list[dict]
) -> list[dict]:
    """
    Run deterministic SQL over all employee rows.

    Groups are AND-ed together.

    Conditions inside each group use:
        AND / OR
    """

    columns = set(
        _column_names()
    )

    group_clauses = []
    params = []

    for group in groups:

        op = group.get(
            "op",
            "and"
        )

        conditions = group.get(
            "conditions",
            []
        ) or []

        parts = []

        for condition in conditions:

            field = condition.get(
                "field"
            )

            match = condition.get(
                "match",
                "eq"
            )

            value = condition.get(
                "value"
            )

            if field not in columns:
                continue

            if value in (
                None,
                ""
            ):
                continue

            if match == "contains":

                parts.append(
                    f'LOWER("{field}") LIKE ?'
                )

                params.append(
                    f"%{str(value).lower()}%"
                )

            else:

                parts.append(
                    f'LOWER("{field}") = ?'
                )

                params.append(
                    str(value).lower()
                )

        if parts:

            joiner = (
                " OR "
                if op == "or"
                else " AND "
            )

            group_clauses.append(
                "("
                + joiner.join(parts)
                + ")"
            )

    # No conditions = no filter.
    sql = "SELECT * FROM employees"

    if group_clauses:

        sql += (
            " WHERE "
            + " AND ".join(
                group_clauses
            )
        )

    log.info(
        "SQL: %s | params=%s",
        sql,
        params
    )

    con = sqlite3.connect(
        config.DB_PATH
    )

    con.row_factory = sqlite3.Row

    rows = [
        dict(row)
        for row in con.execute(
            sql,
            params
        )
    ]

    con.close()

    return rows


# =========================================================
# EXACT NAME LOOKUP
# =========================================================

def exact_name_lookup(
    question: str
) -> list[dict]:
    """
    Deterministically find employee names
    mentioned in the question.

    This avoids counting all rows when
    the router returns exact_lookup without groups.
    """

    question_lower = question.lower()

    con = sqlite3.connect(
        config.DB_PATH
    )

    con.row_factory = sqlite3.Row

    rows = [
        dict(row)
        for row in con.execute(
            "SELECT * FROM employees"
        )
    ]

    con.close()

    matches = []

    for row in rows:

        name = str(
            row.get(
                "name",
                ""
            )
        ).strip()

        if not name:
            continue

        if name.lower() in question_lower:

            matches.append(
                row
            )

    return matches


# =========================================================
# EXTRACT NAME FROM QUESTION
# =========================================================

def extract_person_name(
    question: str
) -> str | None:
    """
    Fallback person-name extractor.

    This is used only if exact_name_lookup
    needs a simple name pattern.
    """

    patterns = [

        r"(?:what is|what's|tell me|show me|get)\s+(.+?)['’]s\s+"
        r"(?:salary|manager|department|position|location|status|email|phone)",

        r"(.+?)\s+salary",

        r"salary\s+of\s+(.+)",

        r"manager\s+of\s+(.+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            name = (
                match.group(1)
                .strip()
                .strip("?")
                .strip()
            )

            if name:
                return name

    return None


# =========================================================
# LOAD CHROMA CHUNKS
# =========================================================

def _load_chunks() -> list[dict]:
    """
    Load all text chunks from Chroma.
    """

    chroma = chromadb.PersistentClient(
        path=config.CHROMA_PATH
    )

    collection = chroma.get_collection(
        config.COLLECTION
    )

    data = collection.get(
        include=[
            "documents"
        ]
    )

    return [
        {
            "id": chunk_id,
            "text": text
        }
        for chunk_id, text in zip(
            data["ids"],
            data["documents"]
        )
    ]


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    question: str,
    k: int = config.TOP_K
) -> list[dict]:
    """
    Vector search + BM25 + RRF.
    """

    chunks = _load_chunks()

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------

    query_embedding = client.embeddings.create(
        model=config.EMBED_MODEL,
        input=[
            question
        ]
    ).data[0].embedding

    chroma = chromadb.PersistentClient(
        path=config.CHROMA_PATH
    )

    collection = chroma.get_collection(
        config.COLLECTION
    )

    vector_results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            k,
            len(texts)
        )
    )

    vector_ids = (
        vector_results["ids"][0]
    )

    # -----------------------------------------------------
    # BM25 SEARCH
    # -----------------------------------------------------

    bm25 = BM25Okapi(
        [
            text.lower().split()
            for text in texts
        ]
    )

    scores = bm25.get_scores(
        question.lower().split()
    )

    bm25_ids = [
        chunks[index]["id"]
        for index in sorted(
            range(
                len(scores)
            ),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
    ]

    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    rrf = {}

    for rank, chunk_id in enumerate(
        vector_ids
    ):

        rrf[chunk_id] = (
            rrf.get(
                chunk_id,
                0
            )
            + 1 / (60 + rank)
        )

    for rank, chunk_id in enumerate(
        bm25_ids
    ):

        rrf[chunk_id] = (
            rrf.get(
                chunk_id,
                0
            )
            + 1 / (60 + rank)
        )

    by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    top_ids = sorted(
        rrf,
        key=rrf.get,
        reverse=True
    )[:k]

    return [
        {
            "text": by_id[chunk_id]["text"],
            "id": chunk_id
        }
        for chunk_id in top_ids
    ]


# =========================================================
# ROW TO TEXT
# =========================================================

def row_to_line(
    row: dict
) -> str:

    return ", ".join(
        f"{key}={value}"
        for key, value in row.items()
        if value
    )


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Grounded LLM answer.
    """

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY from the context. "
                    "Do not invent information. "
                    "Do not guess. "
                    "If the context is empty or does not "
                    "contain the answer, say you don't "
                    "have enough information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# AGGREGATION ANSWER
# =========================================================

def generate_aggregation_answer(
    question: str,
    count: int,
    rows: list[dict]
) -> str:
    """
    Python calculates the exact number.

    Return it directly so the LLM cannot
    add incorrect extra statements.
    """

    return (
        f"There are {count} "
        f"matching employees."
    )


# =========================================================
# LIVE WEATHER
# =========================================================

def live_weather_answer(
    question: str,
    city: str
) -> dict:

    city = (
        city
        or live_data.extract_city(
            question
        )
        or ""
    )

    if city:

        evidence = (
            live_data.get_current_weather(
                city
            )
        )

    else:

        evidence = {
            "error": "No city found."
        }

    if "error" not in evidence:

        context = "\n".join(
            f"{key}: {value}"
            for key, value in evidence.items()
        )

    else:

        context = evidence["error"]

    return {
        "strategy": "live_lookup",
        "answer": generate_answer(
            question,
            context
        ),
        "record_count": None,
        "sources": [
            {
                "type": "live_api",
                "provider": "OpenWeatherMap",
                "city": city
            }
        ],
        "live_evidence": evidence,
        "filters": []
    }


# =========================================================
# EXACT LOOKUP ANSWER
# =========================================================

def exact_lookup_answer(
    question: str
) -> dict:
    """
    Find one employee deterministically
    and answer from that row.
    """

    rows = exact_name_lookup(
        question
    )

    # ---------------------------------------------
    # Fallback using extracted name
    # ---------------------------------------------

    if not rows:

        person_name = extract_person_name(
            question
        )

        if person_name:

            con = sqlite3.connect(
                config.DB_PATH
            )

            con.row_factory = sqlite3.Row

            rows = [
                dict(row)
                for row in con.execute(
                    """
                    SELECT *
                    FROM employees
                    WHERE LOWER(name) = ?
                    """,
                    (
                        person_name.lower(),
                    )
                )
            ]

            con.close()

    if not rows:

        return {
            "strategy": "exact_lookup",
            "answer": (
                "I don't have enough information."
            ),
            "record_count": 0,
            "sources": [],
            "filters": []
        }

    # Only one specific person is expected.
    row = rows[0]

    context = row_to_line(
        row
    )

    answer = generate_answer(
        question,
        context
    )

    return {
        "strategy": "exact_lookup",
        "answer": answer,
        "record_count": 1,
        "sources": [
            row
        ],
        "filters": []
    }


# =========================================================
# PDF FALLBACK RETRIEVAL
# =========================================================

def pdf_fallback_answer(
    question: str
) -> dict:
    """
    Search the currently indexed PDF.

    This is used when the router chooses exact_lookup
    but the employee database has no matching record.
    """

    hits = hybrid_search(
        question
    )

    context = "\n".join(
        hit["text"]
        for hit in hits
    )

    if not context:

        return {
            "strategy": "hybrid_retrieval",
            "answer": (
                "I don't have enough information."
            ),
            "record_count": 0,
            "sources": [],
            "filters": []
        }

    return {
        "strategy": "hybrid_retrieval",
        "answer": generate_answer(
            question,
            context
        ),
        "record_count": None,
        "sources": hits,
        "filters": []
    }


# =========================================================
# MAIN QUERY FUNCTION
# =========================================================

def query_document(
    question: str
) -> dict:
    """
    Main entry point.

    Question
       ↓
    Router
       ↓
    Strategy
       ↓
    Tool / Retrieval
       ↓
    Grounded Answer
    """

    question = question.strip()

    if not question:

        return {
            "strategy": "",
            "answer": "Please enter a question.",
            "record_count": 0,
            "sources": [],
            "filters": []
        }

    # -----------------------------------------------------
    # ROUTER
    # -----------------------------------------------------

    route = classify_query(
        question
    )

    strategy = route[
        "strategy"
    ]

    groups = route[
        "groups"
    ]

    log.info(
        "Strategy: %s | groups=%s",
        strategy,
        groups
    )

    # -----------------------------------------------------
    # LIVE LOOKUP
    # -----------------------------------------------------

    if strategy == "live_lookup":

        return live_weather_answer(
            question,
            route["city"]
        )

    # -----------------------------------------------------
    # EXACT LOOKUP
    # -----------------------------------------------------

    if strategy == "exact_lookup":

        employee_result = exact_lookup_answer(
            question
        )

        # Employee DB has the requested record.
        if employee_result["record_count"] > 0:

            return employee_result

        # No employee record found.
        # Search the active uploaded PDF instead.
        log.info(
            "Exact lookup returned no employee record; "
            "falling back to PDF hybrid retrieval."
        )

        return pdf_fallback_answer(
            question
        )

    # -----------------------------------------------------
    # STRUCTURED / AGGREGATION
    # -----------------------------------------------------

    if strategy in (
        "structured_filter",
        "aggregation"
    ):

        rows = apply_filters(
            groups
        )

        # ---------------------------------------------
        # Aggregation
        # ---------------------------------------------

        if strategy == "aggregation":

            count = len(rows)

            return {
                "strategy": strategy,
                "answer": generate_aggregation_answer(
                    question,
                    count,
                    rows
                ),
                "record_count": count,
                "sources": rows[:10],
                "filters": groups
            }

        # ---------------------------------------------
        # Structured filter
        # ---------------------------------------------

        context = "\n".join(
            row_to_line(row)
            for row in rows
        )

        if not context:

            context = "(no matching rows)"

        return {
            "strategy": strategy,
            "answer": generate_answer(
                question,
                context
            ),
            "record_count": len(rows),
            "sources": rows[:10],
            "filters": groups
        }

    # -----------------------------------------------------
    # HYBRID RETRIEVAL
    # -----------------------------------------------------

    hits = hybrid_search(
        question
    )

    context = "\n".join(
        hit["text"]
        for hit in hits
    )

    if not context:

        context = "(nothing retrieved)"

    return {
        "strategy": "hybrid_retrieval",
        "answer": generate_answer(
            question,
            context
        ),
        "record_count": None,
        "sources": hits,
        "filters": []
    }


# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":

    test_question = (
        "How many employees are On Leave?"
    )

    result = query_document(
        test_question
    )

    print(
        "Strategy:",
        result["strategy"]
    )

    print(
        "Count:",
        result["record_count"]
    )

    print(
        "Answer:",
        result["answer"]
    )