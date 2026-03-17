import re
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .db import get_db, get_llm, get_sql_chain, get_answer_chain, get_clarify_chain, run_query
from .hybrid_search import index_products, hybrid_context
from .charts import should_generate_chart, generate_chart_config
from .evaluation import evaluate_rag

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Indexing products into ChromaDB...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, index_products)
    print("[Startup] Ready.")
    yield


app = FastAPI(
    title="E-commerce RAG Over SQL",
    description="Natural language queries over MySQL database",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False
    ground_truth: str | None = None


class QueryResponse(BaseModel):
    question: str
    sql_query: str
    answer: str
    chart_config: dict | None = None
    clarification_needed: bool = False
    clarification_question: str | None = None
    evaluation: dict | None = None
    sql_results: list[dict] = []


def clean_sql(raw: str) -> str:
    """Strip markdown, prefix text, and semicolons from LLM SQL output."""
    # Remove markdown fences
    cleaned = re.sub(r"```sql|```", "", raw).strip()
    # Remove "Question: ..." and "SQLQuery:" prefix lines that Groq sometimes adds
    cleaned = re.sub(r"(?i)^(question|sqlquery|sql query)\s*:.*\n?", "", cleaned, flags=re.MULTILINE).strip()
    # Remove leading SQLQuery: on same line as SELECT
    cleaned = re.sub(r"(?i)^sqlquery\s*:", "", cleaned).strip()
    # Remove trailing semicolon
    cleaned = cleaned.rstrip(";").strip()
    return cleaned


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        db = get_db()
        llm = get_llm()

        # Step 1: Clarification check
        clarify_chain = get_clarify_chain(llm)
        clarify_result = clarify_chain.invoke({"question": req.question}).strip()
        if clarify_result.startswith("CLARIFY:"):
            clarify_question = clarify_result.replace("CLARIFY:", "").strip()
            return QueryResponse(
                question=req.question,
                sql_query="",
                answer="",
                clarification_needed=True,
                clarification_question=clarify_question
            )

        # Step 2: NL → SQL
        sql_chain = get_sql_chain(db, llm)
        raw_sql = sql_chain.invoke({"question": req.question})
        sql_query = clean_sql(raw_sql)

        # Step 3: Execute SQL
        try:
            sql_results = run_query(db, sql_query)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SQL error: {str(e)}\n\nGenerated SQL:\n{sql_query}")

        # Step 4: Hybrid context (SQL + vector)
        context = hybrid_context(req.question, sql_results)

        # Step 5: Generate answer
        answer_chain = get_answer_chain(llm)
        answer = answer_chain.invoke({
            "question": req.question,
            "sql_query": sql_query,
            "query_results": context
        })

        # Step 6: Chart generation
        chart_config = None
        if should_generate_chart(req.question, llm) and sql_results:
            chart_config = generate_chart_config(req.question, sql_results, llm)

        # Step 7: RAGAS evaluation (optional)
        evaluation = None
        if req.evaluate:
            evaluation = evaluate_rag(
                question=req.question,
                answer=answer,
                contexts=[context],
                ground_truth=req.ground_truth,
            )

        return QueryResponse(
            question=req.question,
            sql_query=sql_query,
            answer=answer,
            chart_config=chart_config,
            evaluation=evaluation,
            sql_results=sql_results[:20]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/sample-questions")
async def sample_questions():
    return {
        "questions": [
            "Which customers placed the most orders?",
            "Show me total revenue by product category as a chart",
            "What are the top 5 best-rated products?",
            "How many orders were cancelled or returned?",
            "Which city has the most customers?",
            "What is the average order value per customer tier?",
            "Show me monthly revenue trend as a line chart",
            "Which products have stock below 100 units?",
            "Who are our Platinum tier customers?",
            "What are the most reviewed products?",
        ]
    }