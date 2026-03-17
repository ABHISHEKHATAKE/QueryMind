"""
db.py — MySQL connection + LangChain SQL chain (using ChatGroq)
"""
import os
import time
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── Connection ──────────────────────────────────────────────────────────────

def get_engine():
    db_user = os.getenv("DB_USER", "root")
    db_password = quote(os.getenv("DB_PASSWORD", ""), safe="")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "ecommerce_rag")
    url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url)


def get_db() -> SQLDatabase:
    engine = get_engine()
    return SQLDatabase(engine, sample_rows_in_table_info=3)


# ── LLMs — two models to avoid rate limits ──────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatGroq:
    """70b model — best SQL quality."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )

def get_fast_llm(temperature: float = 0.0) -> ChatGroq:
    """8b model — faster, higher rate limit. Used for non-SQL calls."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )


# ── Custom SQL Chain ────────────────────────────────────────────────────────

SQL_PROMPT = PromptTemplate.from_template(
    """You are a MySQL expert. Given the database schema and a question, write a valid MySQL SELECT query.

Database Schema:
{schema}

Rules:
- Return ONLY the raw SQL query, nothing else
- No markdown, no explanation, no "SQLQuery:" prefix, no backticks around the whole query
- No semicolon at the end
- Use proper MySQL syntax
- For "top N" questions use LIMIT 5 if N is not specified

Question: {question}

SQL:"""
)

def get_sql_chain(db: SQLDatabase, llm: ChatGroq):
    schema = db.get_table_info()
    prompt = SQL_PROMPT.partial(schema=schema)
    return prompt | llm | StrOutputParser()


# ── Execute SQL ─────────────────────────────────────────────────────────────

def run_query(db: SQLDatabase, sql: str) -> list[dict]:
    import decimal
    import datetime
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = result.fetchall()
    def serialize(val):
        if isinstance(val, decimal.Decimal):
            return float(val)
        if isinstance(val, (datetime.datetime, datetime.date)):
            return str(val)
        return val
    return [
        {col: serialize(val) for col, val in zip(cols, row)}
        for row in rows
    ]


# ── Answer Chain ────────────────────────────────────────────────────────────

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful e-commerce data analyst assistant.

Given the user's question, the SQL query, and the query results,
provide a clear, concise, and friendly natural language answer.

If results are empty, say so clearly.
If the question asks for a chart, mention that a chart is being generated.

User Question: {question}
SQL Query: {sql_query}
Query Results: {query_results}

Answer:"""
)

def get_answer_chain(llm: ChatGroq):
    return ANSWER_PROMPT | llm | StrOutputParser()


# ── Clarification Chain ─────────────────────────────────────────────────────

CLARIFY_PROMPT = PromptTemplate.from_template(
    """You are an e-commerce data assistant. Decide if this question can be answered from a database.

Question: {question}

Available tables: customers, products, orders, order_items, categories, reviews

Rules:
- Default to CLEAR for almost everything
- CLEAR examples: "how many customers", "top products", "total revenue", "cancelled orders", "tell me about orders"
- Only reply CLARIFY if the question has absolutely no SQL interpretation
- Never ask for clarification about quantity or time period
- When in doubt, reply CLEAR

Reply with exactly one of:
CLEAR
or
CLARIFY: <one short question>

Response:"""
)

def get_clarify_chain(llm: ChatGroq):
    return CLARIFY_PROMPT | llm | StrOutputParser()