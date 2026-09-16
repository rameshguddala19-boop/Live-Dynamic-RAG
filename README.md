# Advanced RAG + Live Data

A production-style Advanced RAG system that routes each question to the right strategy instead of always using top-k retrieval.

The system combines SQL, Python aggregation, exact lookup, hybrid retrieval, and live weather data.

## Architecture

```text
Question
   |
   v
classify_query()
   |
   +-------------------+-------------------+-------------------+-------------------+
   |                   |                   |                   |                   |
   v                   v                   v                   v                   v
Structured Filter   Aggregation       Exact Lookup      Hybrid Retrieval      Live Lookup
      SQL            Python            One record       BM25 + Vector + RRF   Weather API
   |                   |                   |                   |                   |
   +-------------------+-------------------+-------------------+-------------------+
                                       |
                                       v
                              generate_answer()
                                       |
                                       v
                                  Final Answer
```

## Key Features

- SQL-based structured filtering
- Python-verified aggregation
- Exact record lookup
- Hybrid retrieval using BM25 + vector search + RRF
- Live weather lookup using OpenWeatherMap
- Grounded answer generation
- Independent evaluation harness
- RAGAS evaluation

## Data Flow

```text
Employee PDF
     |
     v
prepare_data.py
     |
     +-------------------+
     |                   |
     v                   v
  SQLite             ChromaDB
     |                   |
     +---------+---------+
               |
               v
       classify_query()
               |
               v
        Select strategy
               |
               v
       generate_answer()
               |
               v
          Final Answer
```

## Project Structure

```text
Live-Dynamic-RAG/
│
├── app.py
├── advanced_rag.py
├── prepare_data.py
├── config.py
├── live_data.py
├── evaluate_rag.py
├── requirements.txt
├── README.md
├── eval_report.csv
│
├── data/
│   └── Employee_Details_100_1.pdf
│
├── rag.db
└── chroma_store/
```

## Tech Stack

- Python 3.12
- OpenAI API
- SQLite
- ChromaDB
- BM25
- Streamlit
- RAGAS
- OpenWeatherMap API

## Setup

Create and activate a Python 3.12 virtual environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

## Data Ingestion

Place the PDF inside the `data/` folder and run:

```bash
python prepare_data.py
```

The PDF is converted into:

- SQLite rows for exact filtering and counting
- Chroma embeddings for semantic retrieval

## Run the Application

```bash
streamlit run app.py
```

## Evaluation

Fast evaluation:

```bash
python evaluate_rag.py
```

RAGAS evaluation:

```bash
python evaluate_rag.py --with-ragas
```

## Evaluation Results

### Router Accuracy

```text
9/9
```

### Filter / Count Accuracy

```text
5/5
```

### Live City Accuracy

```text
2/2
```

### RAGAS

```text
Faithfulness:      1.0000
Answer Relevancy:  0.8839
Context Recall:    1.0000
```

## Reliability Design

### 1. The LLM does not count

For aggregation questions, Python computes the verified count. The LLM only phrases the verified result.

### 2. Filters use groups

The system supports compound logic such as:

```text
(Finance OR IT) AND Bengaluru
```

Each group has its own AND/OR logic, while the groups are combined with AND.

### 3. Retrieval is not always used

The router decides whether a question should use:

- SQL
- Python aggregation
- Exact lookup
- Hybrid retrieval
- Live API

### 4. One grounding step

All strategies ultimately provide evidence to the same answer-generation step.

## Example Questions

```text
List all employees in Engineering

How many employees are On Leave?

How many employees are in Finance or IT?

How many employees are in Finance or IT and in Bengaluru?

Which employees work in Bengaluru and are Active?

What is the weather in Hyderabad right now?
```

## Conclusion

This project demonstrates a routed Advanced RAG architecture where retrieval is one tool among several.

Structured questions use deterministic SQL and Python logic, open-ended questions use hybrid retrieval, and real-time questions use a live API.