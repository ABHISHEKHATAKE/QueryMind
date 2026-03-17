"""
test_backend.py — Complete backend test suite for RAG over SQL project
Run with: python test_backend.py

Make sure your FastAPI server is running first:
  python run.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
INFO = "\033[94m→\033[0m"
WARN = "\033[93m⚠\033[0m"

results = {"passed": 0, "failed": 0}

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
        results["passed"] += 1
    else:
        print(f"  {FAIL}  {label}")
        if detail:
            print(f"         {detail}")
        results["failed"] += 1

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def query(question, evaluate=False):
    return requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "evaluate": evaluate},
        timeout=60
    )

# ── TEST 1: Health check ─────────────────────────────────────────────────────
section("TEST 1 — Health check")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("Server is running",             r.status_code == 200)
    check("Returns status ok",             r.json().get("status") == "ok")
except Exception as e:
    check("Server is running", False, str(e))
    print(f"\n  {WARN}  Server not reachable. Start it with: python run.py\n")
    exit(1)

# ── TEST 2: Sample questions endpoint ───────────────────────────────────────
section("TEST 2 — Sample questions")
r = requests.get(f"{BASE_URL}/sample-questions")
data = r.json()
check("Endpoint returns 200",             r.status_code == 200)
check("Returns list of questions",        isinstance(data.get("questions"), list))
check("Has at least 5 questions",         len(data.get("questions", [])) >= 5)
print(f"  {INFO}  Sample: '{data['questions'][0]}'")

# ── TEST 3: Simple count query ───────────────────────────────────────────────
section("TEST 3 — Simple count query")
print(f"  {INFO}  Asking: 'How many customers do we have?'")
r = query("How many customers do we have?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer field",                 bool(data.get("answer")))
check("Has sql_query field",              bool(data.get("sql_query")))
check("SQL contains COUNT",               "COUNT" in data.get("sql_query","").upper())
check("SQL contains customers",           "customers" in data.get("sql_query","").lower())
check("No clarification needed",          not data.get("clarification_needed"))
print(f"  {INFO}  Answer: {data.get('answer','')[:80]}...")
print(f"  {INFO}  SQL: {data.get('sql_query','')[:80]}...")

# ── TEST 4: JOIN query ───────────────────────────────────────────────────────
section("TEST 4 — Multi-table JOIN query")
print(f"  {INFO}  Asking: 'Which customers placed the most orders?'")
r = query("Which customers placed the most orders?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("SQL uses JOIN",                    "JOIN" in data.get("sql_query","").upper())
check("SQL uses GROUP BY",                "GROUP BY" in data.get("sql_query","").upper())
check("Has sql_results",                  len(data.get("sql_results", [])) > 0)
check("Results have name field",          "name" in (data.get("sql_results") or [{}])[0])
print(f"  {INFO}  Top result: {data.get('sql_results', [{}])[0]}")

# ── TEST 5: Aggregation query ────────────────────────────────────────────────
section("TEST 5 — Aggregation + SUM query")
print(f"  {INFO}  Asking: 'What is the total revenue from delivered orders?'")
r = query("What is the total revenue from delivered orders?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("SQL uses SUM",                     "SUM" in data.get("sql_query","").upper())
check("SQL filters delivered",            "delivered" in data.get("sql_query","").lower())
check("Answer mentions a number",         any(c.isdigit() for c in data.get("answer","")))
print(f"  {INFO}  Answer: {data.get('answer','')[:100]}...")

# ── TEST 6: Chart generation ─────────────────────────────────────────────────
section("TEST 6 — Chart generation")
print(f"  {INFO}  Asking: 'Show total revenue by category as a chart'")
r = query("Show total revenue by category as a chart")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("Chart config generated",           data.get("chart_config") is not None)
if data.get("chart_config"):
    cfg = data["chart_config"]
    check("Chart has type field",         "type" in cfg)
    check("Chart has data field",         "data" in cfg)
    check("Chart has labels",             bool(cfg.get("data", {}).get("labels")))
    check("Chart has datasets",           bool(cfg.get("data", {}).get("datasets")))
    print(f"  {INFO}  Chart type: {cfg.get('type')}")
    print(f"  {INFO}  Labels: {cfg.get('data',{}).get('labels', [])[:4]}")
else:
    print(f"  {WARN}  No chart generated — chart detection may not have triggered")

# ── TEST 7: Clarification trigger ────────────────────────────────────────────
section("TEST 7 — Clarification for vague question")
print(f"  {INFO}  Asking: 'tell me about orders'  (intentionally vague)")
r = query("tell me about orders")
data = r.json()
check("Returns 200",                      r.status_code == 200)
# This may or may not trigger depending on GPT-4o — both are acceptable
if data.get("clarification_needed"):
    check("Clarification triggered",      True)
    check("Has clarification question",   bool(data.get("clarification_question")))
    print(f"  {INFO}  Clarification asked: {data.get('clarification_question')}")
else:
    print(f"  {WARN}  Clarification not triggered (GPT-4o treated it as clear)")
    check("Has answer anyway",            bool(data.get("answer")))

# ── TEST 8: Status filter query ──────────────────────────────────────────────
section("TEST 8 — WHERE clause / filter query")
print(f"  {INFO}  Asking: 'How many orders were cancelled?'")
r = query("How many orders were cancelled?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("SQL filters cancelled",            "cancel" in data.get("sql_query","").lower())
check("Answer contains a number",         any(c.isdigit() for c in data.get("answer","")))
print(f"  {INFO}  Answer: {data.get('answer','')[:80]}...")

# ── TEST 9: Product / inventory query ───────────────────────────────────────
section("TEST 9 — Product stock query")
print(f"  {INFO}  Asking: 'Which products have stock below 100?'")
r = query("Which products have stock below 100?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("SQL queries products table",       "products" in data.get("sql_query","").lower())
check("SQL has stock condition",          "stock" in data.get("sql_query","").lower() or "100" in data.get("sql_query",""))
check("Has sql_results",                  len(data.get("sql_results", [])) > 0)
print(f"  {INFO}  Found {len(data.get('sql_results', []))} products")

# ── TEST 10: Top N ranking query ─────────────────────────────────────────────
section("TEST 10 — Top N ranking query")
print(f"  {INFO}  Asking: 'What are the top 3 highest rated products?'")
r = query("What are the top 3 highest rated products?")
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
check("SQL uses ORDER BY",                "ORDER BY" in data.get("sql_query","").upper())
check("SQL uses LIMIT",                   "LIMIT" in data.get("sql_query","").upper())
check("Returns max 3 results",            len(data.get("sql_results", [])) <= 3)
print(f"  {INFO}  Results: {[r.get('name','') for r in data.get('sql_results',[])]}")

# ── TEST 11: Response time ───────────────────────────────────────────────────
section("TEST 11 — Response time")
print(f"  {INFO}  Measuring response time for a simple query...")
start = time.time()
r = query("How many products do we have?")
elapsed = round(time.time() - start, 2)
check("Returns valid response",           r.status_code == 200)
check(f"Responds within 30 seconds ({elapsed}s)", elapsed < 30)
if elapsed < 10:
    print(f"  {INFO}  Fast response: {elapsed}s")
elif elapsed < 20:
    print(f"  {WARN}  Moderate response: {elapsed}s (normal for first call)")
else:
    print(f"  {WARN}  Slow response: {elapsed}s (check your OpenAI API latency)")

# ── TEST 12: RAGAS evaluation ────────────────────────────────────────────────
section("TEST 12 — RAGAS evaluation (optional, takes ~30s)")
print(f"  {INFO}  Running with evaluate=True ...")
r = query("What are the top 5 best rated products?", evaluate=True)
data = r.json()
check("Returns 200",                      r.status_code == 200)
check("Has answer",                       bool(data.get("answer")))
if data.get("evaluation"):
    ev = data["evaluation"]
    check("Has faithfulness score",       ev.get("faithfulness") is not None)
    check("Has answer_relevancy score",   ev.get("answer_relevancy") is not None)
    check("Has context_precision score",  ev.get("context_precision") is not None)
    check("Faithfulness > 0.5",           (ev.get("faithfulness") or 0) > 0.5)
    print(f"\n  {INFO}  RAGAS Scores:")
    print(f"         Faithfulness:      {round((ev.get('faithfulness') or 0)*100)}%")
    print(f"         Answer Relevancy:  {round((ev.get('answer_relevancy') or 0)*100)}%")
    print(f"         Context Precision: {round((ev.get('context_precision') or 0)*100)}%")
else:
    print(f"  {WARN}  Evaluation not returned — RAGAS may have failed silently")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
total = results["passed"] + results["failed"]
print(f"\n{'═'*55}")
print(f"  RESULTS: {results['passed']}/{total} tests passed", end="")
if results["failed"] == 0:
    print(f"  \033[92m All tests passed!\033[0m")
elif results["failed"] <= 2:
    print(f"  \033[93m Minor issues\033[0m")
else:
    print(f"  \033[91m Needs attention\033[0m")
print(f"{'═'*55}\n")