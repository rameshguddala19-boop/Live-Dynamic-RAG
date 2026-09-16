import json
import logging
import re
import sqlite3
from pathlib import Path

import chromadb
import pdfplumber
from openai import OpenAI

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s"
)

log = logging.getLogger("prepare")

client = OpenAI()


def _clean_col(name: str, idx: int) -> str:
    c = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    return c if c else f"col_{idx}"


def load_rows(pdf_path: str) -> list[dict]:
    rows = []
    header = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:

                    cells = [(c or "").strip() for c in raw]

                    if not any(cells):
                        continue

                    if header is None:
                        header = [
                            _clean_col(h, i)
                            for i, h in enumerate(cells)
                        ]
                        log.info("Header: %s", header)
                        continue

                    if [
                        _clean_col(c, i)
                        for i, c in enumerate(cells)
                    ] == header:
                        continue

                    rows.append(dict(zip(header, cells)))

    log.info("Extracted %d rows", len(rows))

    return rows


def build_sqlite(rows: list[dict]) -> None:

    if not rows:
        raise ValueError(
            "No rows extracted — check the PDF path / table detection."
        )

    cols = list(rows[0].keys())

    col_defs = ", ".join(
        f'"{c}" TEXT'
        for c in cols
    )

    con = sqlite3.connect(config.DB_PATH)

    con.execute("DROP TABLE IF EXISTS employees")

    con.execute(
        f"CREATE TABLE employees ({col_defs})"
    )

    placeholders = ", ".join(
        "?"
        for _ in cols
    )

    con.executemany(
        f"INSERT INTO employees VALUES ({placeholders})",
        [
            tuple(r.get(c, "") for c in cols)
            for r in rows
        ]
    )

    con.commit()
    con.close()

    log.info(
        "SQLite ready: %s (%d cols)",
        config.DB_PATH,
        len(cols)
    )


def row_to_text(r: dict) -> str:
    return " | ".join(
        f"{k}: {v}"
        for k, v in r.items()
        if v
    )


def build_chroma(rows: list[dict]) -> None:

    texts = [
        row_to_text(r)
        for r in rows
    ]

    resp = client.embeddings.create(
        model=config.EMBED_MODEL,
        input=texts
    )

    vectors = [
        d.embedding
        for d in resp.data
    ]

    chroma = chromadb.PersistentClient(
        path=config.CHROMA_PATH
    )

    try:
        chroma.delete_collection(
            config.COLLECTION
        )
    except Exception:
        pass

    coll = chroma.create_collection(
        config.COLLECTION
    )

    coll.add(
        ids=[
            f"emp-{i}"
            for i in range(len(texts))
        ],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {"json": json.dumps(r)}
            for r in rows
        ]
    )

    log.info(
        "Chroma ready: %d chunks",
        len(texts)
    )


def ingest_pdf(pdf_path: str) -> int:

    rows = load_rows(pdf_path)

    build_sqlite(rows)

    build_chroma(rows)

    return len(rows)


if __name__ == "__main__":

    pdf = "data/Employee_Details_100_1.pdf"

    if not Path(pdf).exists():
        raise SystemExit(
            f"Put the PDF at {pdf} first."
        )

    n = ingest_pdf(pdf)

    print(
        f"Ingest complete. {n} rows."
    )
