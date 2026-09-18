import argparse
import asyncio
import json
import logging
import os
import pathlib
import random
from datetime import datetime, timezone

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from sievrant import SievrantClient

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

SIEVRANT_BASE_URL = os.getenv("SIEVRANT_BASE_URL", "http://localhost:8080")
SIEVRANT_API_KEY = os.getenv("SIEVRANT_API_KEY", "sk-test")

def download_dataset(dataset_name="scifact"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def run_baseline(corpus, queries, qrels, model_name):
    logger.info("Running Baseline Dense Retrieval (FAISS)")
    model = DRES(models.SentenceBERT(model_name), batch_size=64)
    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg["NDCG@10"], recall["Recall@100"]

async def run_sievrant(corpus, queries, qrels, corpus_name, skip_ingest=False):
    logger.info(f"Running Sievrant Hybrid Retrieval on corpus: {corpus_name}")
    async with SievrantClient(base_url=SIEVRANT_BASE_URL, api_key=SIEVRANT_API_KEY) as client:
        if not skip_ingest:
            logger.info("Ingesting corpus into Sievrant...")
            docs_to_ingest = []
            for doc_id, doc in corpus.items():
                docs_to_ingest.append({
                    "id": doc_id,
                    "content": f"{doc.get('title', '')} {doc.get('text', '')}".strip(),
                    "metadata": {"title": doc.get('title', '')}
                })
            
            # Batch ingest
            batch_size = 500
            for i in range(0, len(docs_to_ingest), batch_size):
                batch = docs_to_ingest[i:i + batch_size]
                await client.ingest(corpus=corpus_name, documents=batch)
                logger.info(f"Ingested {i + len(batch)} / {len(docs_to_ingest)} documents")

        logger.info("Querying Sievrant...")
        results = {}
        # queries is dict: {query_id: query_text}
        total_queries = len(queries)
        for idx, (query_id, query_text) in enumerate(queries.items()):
            if idx % 50 == 0:
                logger.info(f"Processed {idx} / {total_queries} queries")
            
            packet = await client.query(
                corpus=corpus_name,
                query=query_text,
                coverage=0.95,
            )
            
            doc_scores = {}
            for chunk in packet.evidence:
                doc_scores[chunk.source_doc] = chunk.rerank_score or chunk.rrf_score
            
            results[query_id] = doc_scores

        logger.info("Evaluating Sievrant results...")
        retriever = EvaluateRetrieval()
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        return ndcg["NDCG@10"], recall["Recall@100"]

def print_results_table(dataset, model, b_ndcg, b_rec, s_ndcg, s_rec):
    print("\n" + "=" * 60)
    print(f"Sievrant BEIR Benchmark: {dataset}")
    print(f"Model: {model}")
    print("=" * 60)
    print()
    print(f"| System             | NDCG@10 | Recall@100 |")
    print(f"|--------------------|---------|------------|")
    print(f"| Dense only (FAISS) | {b_ndcg:7.3f} | {b_rec:10.3f} |")
    print(f"| Sievrant Hybrid    | {s_ndcg:7.3f} | {s_rec:10.3f} |")
    
    ndcg_delta = ((s_ndcg - b_ndcg) / b_ndcg) * 100 if b_ndcg > 0 else 0
    rec_delta = ((s_rec - b_rec) / b_rec) * 100 if b_rec > 0 else 0
    
    print(f"| Delta              | {ndcg_delta:+6.1f}% | {rec_delta:+9.1f}% |")
    print("\nResults saved to disk.")

async def main():
    parser = argparse.ArgumentParser(description="BEIR benchmark for Sievrant")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset name")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Sentence-transformers model")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, run queries only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to write results JSON")
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    corpus, queries, qrels = download_dataset(args.dataset)
    
    b_ndcg, b_rec = run_baseline(corpus, queries, qrels, args.model)
    
    s_ndcg, s_rec = await run_sievrant(corpus, queries, qrels, f"benchmark_{args.dataset}", args.skip_ingest)
    
    print_results_table(args.dataset, args.model, b_ndcg, b_rec, s_ndcg, s_rec)

    if args.results_dir:
        out_dir = pathlib.Path(args.results_dir)
    else:
        out_dir = pathlib.Path(__file__).parent.absolute() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    out_file = out_dir / f"{args.dataset}_{date_str}.json"
    config_file = out_dir / f"{args.dataset}_{date_str}_config.json"

    result_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "timestamp": timestamp,
        "baseline": {"NDCG@10": b_ndcg, "Recall@100": b_rec},
        "sievrant": {"NDCG@10": s_ndcg, "Recall@100": s_rec},
    }
    config_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "timestamp": timestamp,
        "sievrant_base_url": SIEVRANT_BASE_URL,
        "skip_ingest": args.skip_ingest,
    }

    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)
    with open(config_file, "w") as f:
        json.dump(config_payload, f, indent=2)
    logger.info(f"Results written to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
