# rag-engineering-fundamentals

End-to-end RAG pipeline implementation covering vector search with ChromaDB and Pinecone, metadata filtering, re-ranking, and RAGAS evaluation. Built as part of an AI engineering learning sprint.

---

## What's in here

| File/Folder | Description |
|---|---|
| `rag_chatbot/` | Basic RAG chatbot with streaming responses and LangSmith tracing |
| `pinecone_demo/` | Pinecone index setup, metadata filtering, similarity search, and re-ranking |
| `ragas_eval/` | RAGAS testset generation and evaluation pipeline |

---

## Stack

- **LangChain** — RAG pipeline orchestration, LCEL chains, agent construction
- **ChromaDB** — Local vector store with persistent storage
- **Pinecone** — Cloud vector store with namespace isolation and metadata filtering
- **OpenAI** — Embeddings (`text-embedding-3-small`) and completions (`gpt-4o-mini`)
- **RAGAS** — Automated RAG evaluation: testset generation and metric scoring
- **LangSmith** — Tracing and observability for LLM calls
- **Jupyter Notebooks** — Exploratory evaluation work

---

## RAG Chatbot

A document Q&A pipeline over a GitHub repository of `.md` files. Key features:

- Full RAG pipeline: load → chunk → embed → store → retrieve → generate
- Streaming responses via LangChain LCEL
- LangSmith tracing for full call visibility
- Agentic retrieval: LLM decides adaptively what to retrieve based on context

**Architecture:**

```
User query
    │
    ▼
LLM decides whether/what to retrieve
    │
    ▼
Vector similarity search → top-k chunks
    │
    ▼
Prompt assembly (system prompt + context + query)
    │
    ▼
LLM generates grounded response
```

---

## Pinecone Demo

Demonstrates the full Pinecone retrieval stack:

- Index creation with dimension and metric config matching the embedding model
- Namespace-based partitioning for logical document isolation
- Metadata filtering at the field level within namespaces
- Similarity search returning scores
- Re-ranking with a cross-encoder to improve result ordering

**Query execution hierarchy:**

```
1. Target namespace      → coarse partition (deterministic)
2. Metadata filter       → field-level scoping within partition (deterministic)
3. Similarity search     → semantic ranking of remaining vectors (probabilistic)
4. Score threshold       → minimum relevance cutoff
5. Re-ranker             → secondary ordering pass on top-k results
6. Return top k          → final context passed to LLM
```

Hard constraints (namespace, metadata) execute before probabilistic search. By the time similarity runs, it operates on a small, highly relevant subset of the index.

---

## RAGAS Evaluation

Automated evaluation pipeline built on the RAGAS framework.

**Testset generation:**

```python
generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

testset = generator.generate_with_langchain_docs(
    docs,
    testset_size=10,
    query_distribution=[
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
    ]
)
```

**Metrics:**

| Metric | What it measures | Component |
|---|---|---|
| Context Precision | Are relevant chunks ranked at the top? | Retriever |
| Context Recall | Were all necessary chunks retrieved? | Retriever |
| Faithfulness | Did the LLM stick to the context (no hallucination)? | Generator |
| Answer Relevancy | Did the response actually answer the question? | Generator |

**Diagnostic mapping:**

```
Low Context Precision  → noisy retrieval, fix: lower k, add metadata filters, re-ranking
Low Context Recall     → missing chunks, fix: increase k, better chunking, embedding model
Low Faithfulness       → LLM hallucinating beyond context, fix: tighten system prompt
Low Answer Relevancy   → off-topic or padded responses, fix: prompt engineering
```

---

## Setup

```bash
git clone https://github.com/saikaushik1997/rag-engineering-fundamentals.git
cd rag-engineering-fundamentals
pip install langchain langchain-openai chromadb pinecone-client ragas langsmith
```

Set environment variables:

```bash
OPENAI_API_KEY=...
PINECONE_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

---

## Learnings

### Docstrings are runtime behavior, not documentation

In traditional software engineering, docstrings are cosmetic — you could delete every one and the program runs identically. In AI engineering, the docstring is the only information the LLM has about what a tool does. It reads that description at runtime and makes a decision about whether to call it.

A vague docstring is a bug. A well-written one is a prompt. The same principles that apply to system prompts apply to tool docstrings: be explicit, be specific, describe both what it does and *when* to use it. With multiple tools available, unclear docstrings cause the LLM to pick the wrong one consistently.

This extends to a broader point: anything an LLM reads at runtime is functional, not cosmetic. Variable names, system prompt wording, retrieval constraints — these are all code, just interpreted by a probabilistic system instead of a deterministic compiler.

### Namespace, metadata filtering, and similarity search are three independent layers

An intuitive assumption is that metadata filtering happens at the namespace level — but they operate at different granularities and stack on top of each other:

```
Namespace    → which partition to search (like which library)
Metadata     → which documents within that partition (like which shelf)
Similarity   → which documents on that shelf are most relevant (like which book)
```

A query applies namespace scoping first, then metadata filtering, and only then runs the similarity search — on whatever subset remains. By the time the search algorithm kicks in, it's working against a much smaller, more relevant candidate set.

### Pre-filtering vs post-filtering is about what the search sees, not what gets eliminated

Both approaches eliminate the same vectors that don't match metadata. The difference is *when* the filter runs relative to the similarity search:

- **Pre-filter**: search runs on a restricted set → guaranteed result count, potentially weaker matches
- **Post-filter**: search runs on full index → higher quality matches, unpredictable result count after filtering

The fix for post-filtering's result count problem is over-fetching — retrieve `top_k=20`, then filter, so the filter losses get absorbed and you still return meaningful results.

### Vector stores always return results — score thresholds are a required second defense layer

Similarity search will always return the top-k most similar vectors even if none of them are actually relevant. Without a score threshold, the LLM receives noise and either hallucinates an answer or confabulates details from training data.

Witnessed this directly: a prompt constrained to retrieved context but without a score floor produced a response with details like "ensure packaging remains intact" — plausible-sounding customer service language from training data, completely absent from the retrieved chunk. The retrieval worked; the prompt alone wasn't enough.

Defense in depth for production:

```
Metadata filter    → hard, deterministic, infrastructure level
Score threshold    → hard, deterministic, infrastructure level
LLM prompt constraint ("only answer from context") → soft, probabilistic, prompt level
```

Hard constraints belong at the infrastructure level. Prompt-level constraints alone are insufficient.

### RAGAS evaluates whatever retriever you wire up — not a specific vector store

RAGAS doesn't know or care about Pinecone or ChromaDB. It takes `user_input + retrieved_contexts + response + reference` as plain strings and scores them. The `retrieved_contexts` is just a list — it could come from any source.

This makes it a genuine comparison tool: run the same eval pipeline against dense search, then hybrid search, then dense + re-ranking. Same eval, different retrieval inputs, objective delta. That before/after delta is what makes an eval table in a README meaningful rather than decorative.

### Query distribution shapes what weaknesses your eval can detect

The default distribution (50% single-hop, 25% multi-hop abstract, 25% multi-hop specific) is reasonable but not neutral. A system that scores well on single-hop but poorly on multi-hop is telling you something specific: retrieval handles isolated queries but fails when the answer requires synthesizing across multiple chunks. Skewing distribution toward multi-hop when that's a known weakness generates more targeted test cases and measures improvement more precisely.

### Expensive operations should be persisted immediately

```python
# Knowledge graph — expensive to rebuild, save once
kg.save("knowledge_graph.json")

# Testset — same reason, save right after generation
df.to_csv('testset.csv', index=False)
```

Jupyter is the right tool for evaluation work precisely because expensive operations stay in memory across cells. But persistence is the insurance policy — kernel crashes and closed notebooks shouldn't cost 10 minutes of LLM calls.

### The quality/cost/latency triangle applies to every decision

Every architectural choice is a point on this triangle simultaneously:

```
         Quality
           /\
          /  \
         /    \
     Cost ——————— Latency
```

- Increasing `k` → better recall, more tokens, higher cost and latency
- Using `gpt-4o` over `gpt-4o-mini` → better faithfulness, significantly more expensive
- Re-ranking → better context precision, adds a model inference round trip
- Extended thinking → measurably better reasoning, slower and more expensive

A cascade pattern — cheap model by default, expensive model only when confidence is low — is a legitimate production architecture that explicitly navigates this triangle rather than defaulting to the most capable model for everything.
