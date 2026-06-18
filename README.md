# personal_agent

Personal AI assistant deployed locally — a CLI agent with a local knowledge base, web search, and tool-use capabilities.

## Architecture

```
personal_agent/
├── cli/            # REPL, slash commands, display formatting
├── core/           # Agent loop, conversation management, memory compression
├── kb/             # Knowledge base: embedding, ingestion, retrieval
├── tools/          # Tool implementations (search, ingest, list, remove)
├── config.py       # Configuration from environment variables
└── main.py         # Entry point
```

### Knowledge Base (RAG Pipeline)

The KB uses a production-quality hybrid retrieval pipeline:

- **Embeddings**: `BAAI/bge-m3` via FlagEmbedding — 1024-dim dense vectors + sparse lexical weights
- **Chunking**: Semantic sentence-boundary chunking with embedding-similarity merging
- **Retrieval**: Dense search (Chroma top-20) + sparse BM25 search → RRF fusion → cross-encoder reranker (`bge-reranker-v2-m3`) → top-5
- **Query rewriting**: LLM-powered query expansion before retrieval
- **File support**: Markdown, PDF, DOCX, plain text

## Setup

Requires Python 3.12+ and Poetry.

```bash
git clone https://github.com/andy-c-426/personal_agent.git
cd personal_agent
poetry install
```

Set the required environment variables:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

Optional:

```bash
export PERSONAL_AGENT_DIR="$HOME/.personal_agent"  # Default storage location
```

## Usage

```bash
poetry run personal-agent
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/ingest <path>` | Ingest a file or directory into the KB |
| `/kb list` | List all documents in the KB |
| `/kb remove <path>` | Remove a document from the KB |
| `/config` | Show current configuration |
| `/help` | Show available commands |
| `/quit` | Exit |

### Tools

The agent has access to these tools during conversations:

| Tool | Description |
|------|-------------|
| `kb_search` | Hybrid semantic + keyword search with cross-encoder reranking |
| `web_search` | Search the web via Tavily API |
| `kb_ingest` | Add documents to the knowledge base |
| `kb_list` | List indexed documents |
| `kb_remove` | Remove documents from the index |

## Testing

```bash
# Fast tests
poetry run pytest tests/ -v -m "not slow"

# Quality gate (requires model downloads, ~1-2 min)
poetry run pytest tests/test_rag_quality.py -v -s

# All tests
poetry run pytest tests/ -v
```

### Quality Gate

The RAG quality gate (`tests/test_rag_quality.py`) validates retrieval quality with hand-crafted queries:

- **Recall@5** ≥ 0.80 — at least one relevant chunk in top 5
- **MRR** ≥ 0.60 — mean reciprocal rank of first relevant chunk
- **NDCG@5** ≥ 0.70 — ranking quality accounting for multiple relevant chunks

These tests use real models and must pass before any RAG pipeline change.

## Tech Stack

- **Agent**: DeepSeek API via OpenAI client
- **Embeddings**: FlagEmbedding (BAAI/bge-m3)
- **Vector DB**: Chroma (persistent local storage)
- **Reranker**: BGE-Reranker v2-M3 via sentence-transformers
- **CLI**: prompt_toolkit + Rich
- **Web Search**: Tavily API
- **Compression**: tiktoken for token counting

## License

Apache 2.0. See [LICENSE](LICENSE).
