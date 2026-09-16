"""Evaluate the real query_document().

Fast:
    python evaluate_rag.py

With RAGAS:
    python evaluate_rag.py --with-ragas
"""

import argparse
import csv
import logging
import warnings

import advanced_rag as rag

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =========================================================
# INDEPENDENT TEST CASES
# =========================================================

GROUND_TRUTH = [

    {
        "id": "Q1",
        "q": "List all employees in Engineering",
        "strategy": "structured_filter",
        "oracle_groups": [
            {
                "op": "and",
                "conditions": [
                    {
                        "field": "department",
                        "match": "eq",
                        "value": "Engineering",
                    }
                ],
            }
        ],
    },

    {
        "id": "Q3",
        "q": "Which employees work in Bengaluru and are Active?",
        "strategy": "structured_filter",
        "oracle_groups": [
            {
                "op": "and",
                "conditions": [
                    {
                        "field": "location",
                        "match": "contains",
                        "value": "Bengaluru",
                    },
                    {
                        "field": "status",
                        "match": "eq",
                        "value": "Active",
                    },
                ],
            }
        ],
    },

    {
        "id": "Q4",
        "q": "How many employees are On Leave?",
        "strategy": "aggregation",
        "oracle_groups": [
            {
                "op": "and",
                "conditions": [
                    {
                        "field": "status",
                        "match": "eq",
                        "value": "On Leave",
                    }
                ],
            }
        ],
    },

    {
        "id": "Q5",
        "q": "How many employees are in Finance or IT?",
        "strategy": "aggregation",
        "oracle_groups": [
            {
                "op": "or",
                "conditions": [
                    {
                        "field": "department",
                        "match": "eq",
                        "value": "Finance",
                    },
                    {
                        "field": "department",
                        "match": "eq",
                        "value": "IT",
                    },
                ],
            }
        ],
    },

    {
        "id": "Q6",
        "q": "How many employees are in Finance or IT and in Bengaluru?",
        "strategy": "aggregation",
        "oracle_groups": [
            {
                "op": "or",
                "conditions": [
                    {
                        "field": "department",
                        "match": "eq",
                        "value": "Finance",
                    },
                    {
                        "field": "department",
                        "match": "eq",
                        "value": "IT",
                    },
                ],
            },
            {
                "op": "and",
                "conditions": [
                    {
                        "field": "location",
                        "match": "contains",
                        "value": "Bengaluru",
                    }
                ],
            },
        ],
    },

    {
        "id": "LIVE-1",
        "q": "What is the weather in Hyderabad right now?",
        "strategy": "live_lookup",
        "expected_city": "Hyderabad",
    },

    {
        "id": "LIVE-2",
        "q": "What is the weather in Mumbai right now?",
        "strategy": "live_lookup",
        "expected_city": "Mumbai",
    },

    {
        "id": "HYB-1",
        "q": "Give a high-level overview of the Engineering team.",
        "strategy": "hybrid_retrieval",
        "ground_truth": (
            "Engineering employees include their names, positions, "
            "locations and status; the department has 8 people."
        ),
    },

    {
        "id": "HYB-2",
        "q": "What kinds of positions exist in the Engineering department?",
        "strategy": "hybrid_retrieval",
        "ground_truth": (
            "Engineering roles include positions such as Software Engineer "
            "and related technical titles across several locations."
        ),
    },
]


# =========================================================
# ORACLE COUNT
# =========================================================

def oracle_count(row):
    if "oracle_groups" not in row:
        return None

    return len(
        rag.apply_filters(
            row["oracle_groups"]
        )
    )


# =========================================================
# RAGAS
# =========================================================

def run_ragas(hybrid_rows):

    from datasets import Dataset

    from ragas import evaluate
    from ragas.run_config import RunConfig

    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
    )

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from langchain_openai import (
        ChatOpenAI,
        OpenAIEmbeddings,
    )

    import config

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=0,
        )
    )

    embed = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.EMBED_MODEL
        )
    )

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in hybrid_rows
        ]
    )

    return evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=judge,
        embeddings=embed,
        run_config=RunConfig(
            timeout=300,
            max_workers=1,
        ),
    )


# =========================================================
# MAIN EVALUATION
# =========================================================

def main(with_ragas):

    router_hit = 0
    router_total = 0

    count_hit = 0
    count_total = 0

    city_hit = 0
    city_total = 0

    report = []
    hybrid_rows = []

    for row in GROUND_TRUTH:

        result = rag.query_document(
            row["q"]
        )

        # ---------------------------------------------
        # Router accuracy
        # ---------------------------------------------

        router_total += 1

        router_ok = (
            result["strategy"]
            == row["strategy"]
        )

        if router_ok:
            router_hit += 1

        # ---------------------------------------------
        # Count / filter accuracy
        # ---------------------------------------------

        expected_count = oracle_count(
            row
        )

        count_ok = None

        if expected_count is not None:

            count_total += 1

            count_ok = (
                result["record_count"]
                == expected_count
            )

            if count_ok:
                count_hit += 1

        # ---------------------------------------------
        # Live city accuracy
        # ---------------------------------------------

        city_ok = None

        if "expected_city" in row:

            city_total += 1

            actual_city = (
                result.get(
                    "live_evidence",
                    {}
                ).get(
                    "city",
                    ""
                )
            )

            city_ok = (
                actual_city.lower()
                == row["expected_city"].lower()
            )

            if city_ok:
                city_hit += 1

        # ---------------------------------------------
        # Hybrid data for RAGAS
        # ---------------------------------------------

        if (
            row["strategy"]
            == "hybrid_retrieval"
            and "ground_truth" in row
        ):

            contexts = [
                source["text"]
                for source in result["sources"]
                if isinstance(source, dict)
                and "text" in source
            ]

            hybrid_rows.append(
                {
                    "question": row["q"],
                    "answer": result["answer"],
                    "contexts": contexts,
                    "ground_truth": row["ground_truth"],
                }
            )

        # ---------------------------------------------
        # Report row
        # ---------------------------------------------

        report.append(
            {
                "id": row["id"],
                "strategy": result["strategy"],
                "router_ok": router_ok,
                "expected_count": expected_count,
                "actual_count": result["record_count"],
                "count_ok": count_ok,
                "city_ok": city_ok,
                "answer": result["answer"],
            }
        )

    # =====================================================
    # RESULTS
    # =====================================================

    print(
        f"Router accuracy:       "
        f"{router_hit}/{router_total}"
    )

    print(
        f"Filter/count accuracy: "
        f"{count_hit}/{count_total}"
    )

    print(
        f"Live (city) accuracy:  "
        f"{city_hit}/{city_total}"
    )

    # =====================================================
    # RAGAS
    # =====================================================

    if with_ragas and hybrid_rows:

        print(
            "\nRunning RAGAS on the hybrid questions..."
        )

        score = run_ragas(
            hybrid_rows
        )

        print(
            "RAGAS:",
            score
        )

    # =====================================================
    # SAVE CSV
    # =====================================================

    with open(
        "eval_report.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=report[0].keys(),
        )

        writer.writeheader()
        writer.writerows(report)

    print(
        "Wrote eval_report.csv"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="also score hybrid questions with RAGAS",
    )

    args = parser.parse_args()

    main(
        args.with_ragas
    )