"""
evaluation.py — RAGAS evaluation (ChatGroq + HuggingFace embeddings)
"""
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


def evaluate_rag(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Filter out empty contexts
        clean_contexts = [c for c in contexts if c and c.strip()]
        if not clean_contexts:
            clean_contexts = ["No context retrieved."]

        data = {
            "question": [question],
            "answer":   [answer],
            "contexts": [clean_contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Wrap LLM and embeddings for RAGAS
        llm = LangchainLLMWrapper(ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        ))
        embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ))

        metrics = [faithfulness, answer_relevancy, context_precision]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )

        scores = result.to_pandas().iloc[0].to_dict()

        return {
            "faithfulness":      round(float(scores.get("faithfulness")   or 0), 3),
            "answer_relevancy":  round(float(scores.get("answer_relevancy") or 0), 3),
            "context_precision": round(float(scores.get("context_precision") or 0), 3),
            "context_recall":    None,
        }

    except Exception as e:
        # Never crash the main query — return error info as scores
        print(f"[RAGAS] Evaluation failed: {e}")
        return {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_precision": 0.0,
            "context_recall":    None,
            "error":             str(e),
        }