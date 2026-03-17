"""
charts.py — Natural language chart generation (using ChatGroq)
"""
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHART_DETECT_PROMPT = PromptTemplate.from_template(
    """Decide if the following question would benefit from a chart/graph visualization.

Question: {question}

Reply with exactly one word: YES or NO"""
)

CHART_CONFIG_PROMPT = PromptTemplate.from_template(
    """You are a data visualization expert. Convert the query results into a Chart.js configuration object.

Question: {question}
Query Results (list of dicts): {results}

Rules:
- Choose the most appropriate chart type: bar, line, pie, doughnut
- Use clear, readable labels
- Use these colors: ["#6366f1","#06b6d4","#f59e0b","#10b981","#ef4444","#8b5cf6","#f97316"]
- Return ONLY a valid JSON object (no markdown, no explanation) with this structure:
{{
  "type": "bar",
  "data": {{
    "labels": [...],
    "datasets": [{{
      "label": "...",
      "data": [...],
      "backgroundColor": [...]
    }}]
  }},
  "options": {{
    "responsive": true,
    "plugins": {{
      "legend": {{"position": "top"}},
      "title": {{"display": true, "text": "..."}}
    }}
  }}
}}"""
)


def should_generate_chart(question: str, llm: ChatGroq) -> bool:
    chain = CHART_DETECT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()
    return result == "YES"


def generate_chart_config(question: str, results: list[dict], llm: ChatGroq) -> dict | None:
    if not results:
        return None
    chain = CHART_CONFIG_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "results": json.dumps(results[:50])})
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None