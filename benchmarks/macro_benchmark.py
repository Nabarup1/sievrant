import asyncio
import time
from typing import Dict, Any

from sievrant import SievrantClient, EntityExtractor, TreeSummarizer, PromptOptimizer

async def run_benchmark():
    print("========================================")
    print("Sievrant Macro Benchmark (Phase 3.5)")
    print("========================================")
    
    # We use a mocked LLM for benchmarking to avoid network latency/costs
    # In a real environment, set OPENAI_API_KEY to use actual LLMs
    print("\n1. Initializing Subsystems...")
    
    # 1. Start Client
    client = SievrantClient(
        base_url="http://localhost:8080",
        api_key="sk-test"
    )
    
    # 2. Extractors
    extractor = EntityExtractor(model="mock") 
    summarizer = TreeSummarizer(model="mock")
    
    # Ingest document
    print("\n2. Ingesting Baseline Context...")
    start = time.time()
    try:
        await client.ingest(corpus="benchmark-corpus", documents=[
            {"id": "doc1", "content": "Sievrant is a fast RAG framework. It uses Conformal Prediction."},
            {"id": "doc2", "content": "Pinecone is a vector database. It does not handle orchestration."}
        ])
    except Exception as e:
        print(f"Server rejected ingestion (expected for mock benchmark): {e}")
    ingest_time = time.time() - start
    print(f"Ingestion completed in {ingest_time*1000:.2f} ms")
    
    # GraphRAG 
    print("\n3. Executing GraphRAG Entity Extraction...")
    start = time.time()
    extracted = await extractor.extract("doc1:0", "Sievrant is a fast RAG framework. It uses Conformal Prediction.")
    extract_time = time.time() - start
    print(f"Extraction completed in {extract_time*1000:.2f} ms")
    print(f"Entities Found: {len(extracted['entities'])}")
    
    # RAPTOR Summarization
    print("\n4. Executing RAPTOR Tree Summarization...")
    start = time.time()
    node_data = {
        "node_1": [
            "Sievrant is a fast RAG framework.",
            "It uses Conformal Prediction."
        ]
    }
    await summarizer.summarize_nodes("benchmark-corpus", node_data, "sk-test")
    raptor_time = time.time() - start
    print(f"RAPTOR Summarization completed in {raptor_time*1000:.2f} ms")
    
    # Macro Query
    print("\n5. Executing Macro Query (End-to-End)...")
    start = time.time()
    try:
        response = await client.query(
            corpus="benchmark-corpus",
            query="What framework uses Conformal Prediction?", 
            coverage=0.9, 
            token_budget=1000
        )
        query_time = time.time() - start
        print(f"Query completed in {query_time*1000:.2f} ms")
        print(f"Result: {response.answer}")
        print(f"Evidence Used: {response.evidence_count} chunks")
        print(f"Latency: {response.latency_ms} ms (Server-side)")
    except Exception as e:
        # If server is not running, we catch it gracefully
        print(f"Failed to query server: {e}")
        print("Note: Run `cargo run -p sievrant-server` in another terminal.")

    print("\n========================================")
    print("Benchmark Complete")
    print("========================================")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
