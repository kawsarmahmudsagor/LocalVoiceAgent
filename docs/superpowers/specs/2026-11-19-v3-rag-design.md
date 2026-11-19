# Local Voice Bot v3 — RAG and Playground Tabs

- **Date:** 2026-11-19
- **Status:** Draft, awaiting review
- **Branch:** `feat/voice-bot-v3-RAG` (from `feat/voice-bot-v2` at `84c8e37`)
- **Builds on:** [v1 design](2026-11-18-local-voice-bot-design.md) and [v2 design](2026-11-18-v2-model-switching-design.md). v2 §2 (Linux VM, browser audio, `torch==2.7.1+cu128` everywhere) still applies.

## 1. Goal

Let the voice bot answer from the user's own documents, and measure what that costs in latency. Retrieval becomes a timed stage of every voice turn, like STT, LLM and TTS. Two local embedding backends (Hugging Face and Ollama) can be switched from the UI and compared from the turn logs.

The playground also gets horizontal tabs at the top: **Playground**, **System Prompt** and **RAG**.

### What the user asked for

- A RAG system with a vector store in **Chroma**.
- Supported files: **`.txt`, `.md`, `.pdf`, `.docx`**. Several files can be uploaded at once.
- A **RAG tab** with the settings needed to tune RAG.
- A **System Prompt tab**.
- Embedding models from **Hugging Face or Ollama**, switchable in the RAG tab's embedding model section. Defaults: the cheapest reliable model of each.

### Decisions made during design (2026-11-19)

| Topic | Decision | Reason |
|---|---|---|
| Retrieval mode | Retrieve on every turn while RAG is on; inject context only for chunks above a similarity threshold | A voice bot mostly chats. Turns with no match pay only `retrieval_ms`, with no extra prompt tokens |
| Default embedder | Hugging Face `BAAI/bge-small-en-v1.5` (33M params, 384 dims) through sentence-transformers, on CUDA | Best retrieval quality for its size and latency. English only, which matches the English STT models. No extra service |
| Ollama embedder | `nomic-embed-text` (137M params, 768 dims) by default; any other embedding model pulled into Ollama can be picked | The cheapest reliable Ollama embedder |
| Ollama runtime | Docker `ollama/ollama` with the GPU, like llama-server | No `sudo` install on the VM |
| `.doc` (Word 97–2003) | Not supported; `.docx` only | No reliable pure-Python reader. LibreOffice would add about 500 MB and a `sudo` install |
| Chroma mode | Embedded `PersistentClient` in the server process | No extra service; the index survives restarts |

### Success criteria

1. Files of the four types can be uploaded several at a time, are chunked and indexed, and appear in the document list with chunk counts. They can be deleted and re-indexed.
2. With RAG on, a question answered by an uploaded document gets an answer grounded in it, and the reply shows its sources.
3. Every turn's `logs/turns.jsonl` record has `retrieval_ms` and the RAG settings in use, so `latency_report --by rag_embedder,rag_enabled` compares them.
4. The embedder can be switched between Hugging Face and Ollama in the RAG tab while idle, with no restart. A failed switch leaves the previous embedder active.
5. A retrieval failure (Ollama down, embedder error) never fails a turn: the bot answers without context and the UI shows the error.
6. The System Prompt tab edits the system prompt and the RAG context template.
7. Unit tests pass without a GPU, Docker, network or downloaded models.

### Out of scope

- OCR for scanned PDFs: a PDF with no text layer is marked "no text".
- `.doc`, `.rtf`, `.html`, spreadsheets and images.
- Query rewriting from conversation history, hybrid (keyword + vector) search and re-ranking. The query is the current user turn only.
- Several collections or per-conversation document sets: one shared document library.
- Embedding models beyond the two backends, and multilingual retrieval.
- Authentication for the upload API: the server is reached only through the SSH tunnel (v2 §2).

## 2. Latency measurement

New timer stages in `server/voicebot/timing.py`:

| Stage | When |
|---|---|
| `retrieval_start` | the turn starts retrieval (after `stt_done` or `input`) |
| `retrieval_done` | the retriever returned, or failed |

New span `retrieval_ms = retrieval_done - retrieval_start`. It is `null` when RAG is off or the library is empty. `perceived_ms` is unchanged in definition (end of speech to first audio), so it now includes retrieval. `llm_ttft_ms` starts at `retrieval_done` when that mark exists, so it stays the LLM's own time to first token.

Fields added to each `turns.jsonl` record:

| Field | Meaning |
|---|---|
| `rag_enabled` | RAG switch at the start of the turn |
| `rag_embedder` | for example `hf:BAAI/bge-small-en-v1.5`, or `null` when off |
| `rag_top_k`, `rag_min_score`, `rag_max_context_chars` | live settings used |
| `rag_chunk_size`, `rag_chunk_overlap` | index settings of the active collection |
| `rag_chunks_used`, `rag_context_chars` | what was injected |
| `rag_error` | error text, or absent |

The latency panel in the Playground shows `retrieval_ms` after `stt_ms`.

## 3. UI

### 3.1 Tabs

Horizontal tabs across the top of the page: **Playground | System Prompt | RAG**. Switching tabs only hides and shows panels. The WebSocket, the microphone, playback and the conversation keep running, so a turn can continue while another tab is open. The last open tab is remembered in `localStorage` (a missing or blocked value falls back to Playground).

### 3.2 Playground tab

Today's page: chat, composer, status bar, latency panel and settings panel (models, temperature, top P, max tokens, voice). The system prompt textarea moves out to the System Prompt tab. Each assistant reply that used RAG shows a collapsed **Sources** line; expanding it lists the file name, chunk number, similarity score and the first 160 characters of each chunk.

### 3.3 System Prompt tab

- **System prompt:** a large editor for the existing `system_prompt` setting, with **Reset to default**.
- **RAG context template:** the text added to the system message when context is found. It must contain `{context}`. Default:

  ```
  Answer from these notes when they are relevant. If they do not cover the question, answer normally and do not mention them.

  {context}
  ```

  A template without `{context}` is rejected with an inline error, and the previous template stays.

Both are sent with the existing debounced `update_settings` command. `Defaults` gains `rag_template`.

### 3.4 RAG tab

**Embedding model.** A dropdown of the available embedders:

- `hf:BAAI/bge-small-en-v1.5`, always listed.
- `ollama:<model>` for every embedding model in Ollama's `GET /api/tags`, plus `ollama:nomic-embed-text` even before it is pulled (selecting an unpulled model shows "run `scripts/ollama.sh pull nomic-embed-text`"). If Ollama cannot be reached, its entries are shown disabled with "Ollama not reachable at `<OLLAMA_URL>`".

Next to the dropdown: whether the active collection is current ("indexed", "needs re-index", "indexing 3/7"), and its vector size.

**Retrieval (live, apply on the next turn).**

| Setting | Default | Range |
|---|---|---|
| RAG on/off | off | |
| `top_k` | 3 | 1–10 |
| Min similarity (cosine) | 0.5 | 0.0–1.0, step 0.05 |
| Max context characters | 2000 | 200–8000, step 100 |

**Indexing (a change re-indexes every document).**

| Setting | Default | Range |
|---|---|---|
| Chunk size (characters) | 800 | 200–4000 |
| Chunk overlap (characters) | 120 | 0 to half the chunk size |

Changing an indexing setting asks "Re-index all documents?" before applying.

**Documents.**

- Drop zone and file picker accepting several files at once: `.txt`, `.md`, `.pdf`, `.docx`, up to 25 MB each.
- A list with name, type, size, chunk count and state: `queued`, `indexing`, `ready`, `no text`, `failed: <reason>`.
- Delete per document; **Re-index all**.

RAG settings are disabled while a turn is running or the library is indexing, like model switches in v2.

## 4. Server design

### 4.1 Package `server/voicebot/rag/`

| Module | Purpose |
|---|---|
| `loaders.py` | `load_text(path) -> str` by extension. `.txt` and `.md` are read as UTF-8 (falling back to `cp1252`). `.pdf` uses `pypdf` page text joined by blank lines. `.docx` uses `python-docx` paragraphs plus table cell text. An empty result raises `NoText`. Unknown extensions raise `UnsupportedType`. Markdown is indexed as written; heading lines help chunk boundaries. |
| `chunker.py` | `chunk(text, size, overlap) -> list[str]`. Recursive split: blank lines, then line breaks, then sentence ends (`. ! ?`), then spaces, then characters, so no chunk exceeds `size`. Consecutive chunks share `overlap` characters. Whitespace is normalised and empty chunks are dropped. |
| `embedders.py` | `Embedder` protocol: `slug`, `dim`, `embed(texts, kind: "query" \| "document") -> list[list[float]]`, `close()`. `HFEmbedder` loads sentence-transformers on CUDA (CPU if `RAG_EMBED_DEVICE=cpu`), normalises vectors, and adds bge's query prefix `"Represent this sentence for searching relevant passages: "` to queries only. `OllamaEmbedder` calls `POST {OLLAMA_URL}/api/embed` with `{"model", "input": [...]}` and a 10 s timeout, and normalises vectors; `dim` is learned from the first call. `make_embedder(slug)` parses `hf:` and `ollama:`. |
| `store.py` | Chroma `PersistentClient(path=data/rag/chroma, settings=Settings(anonymized_telemetry=False))`. One collection per embedder, named from the slug (`docs-hf-baai-bge-small-en-v1-5`), with `hnsw:space = cosine`. Collection metadata holds `embedder`, `dim`, `chunk_size` and `chunk_overlap`. Embeddings are always passed in, so Chroma never loads its default embedding function or downloads a model. Chunk ids are `<doc_id>:<n>`. Metadata per chunk: `doc_id`, `name`, `chunk`. |
| `library.py` | The document library. Original files go to `data/rag/files/<doc_id><ext>`; `data/rag/manifest.json` lists `doc_id`, name, type, size, sha256, state and chunk count. Settings persist in `data/rag/settings.json`. Uploading a file with the same sha256 as an existing one replaces nothing and reports "already indexed". |
| `service.py` | `RagService`: owns the active embedder, the store and the library. A single indexing worker thread works through a queue of jobs (`index doc`, `delete doc`, `reindex all`). It emits `rag_status` after each step. `retrieve(query) -> Retrieval` runs on the turn thread. |

`data/` is added to `.gitignore` (uploaded documents and the index never go into git).

### 4.2 Retrieval

`retrieve(query)`:

1. Return an empty result if RAG is off or the active collection is empty.
2. Embed the query (`kind="query"`), query Chroma for `top_k` results, and convert cosine distance to similarity (`1 - distance`).
3. Drop results below the minimum similarity. Keep them in rank order until adding the next would exceed the max context characters.
4. Format the context as one block per chunk: `[<name>, part <n>]` followed by the text.

`Retrieval` holds `chunks` (name, chunk number, score, text) and `context` (the formatted string, empty if no chunk qualified).

The embedder and the indexing worker share a lock, so a query never runs on an embedder that is being switched. If indexing is running, a query waits for the current batch (at most one batch of 32 chunks), not the whole job.

### 4.3 Turn flow (`session.py`)

After `stt_done` (or `input` for typed turns) and before the LLM call:

1. Mark `retrieval_start`, call `rag.retrieve(text)` and mark `retrieval_done`.
2. On an exception, log it, send `{"type": "error", "code": "rag_error", "message": ...}`, record `rag_error`, and continue with no context.
3. If `context` is non-empty, `Conversation.messages(text, context=...)` builds the system message as `system_prompt + "\n\n" + rag_template.format(context=context)`. Stored history keeps only the raw user text and the reply; context is never written to history.
4. Send `{"type": "rag_sources", "turn_id", "sources": [{"name", "chunk", "score", "preview"}]}` when at least one chunk was used.

`VoiceSession` takes the `RagService` as an optional dependency. Without it, turns run exactly as before, which keeps the existing tests valid.

### 4.4 HTTP API (`web.py`)

| Route | Behaviour |
|---|---|
| `POST /api/rag/documents` | Multipart, field `files` (one or more). Each file is checked for type and size, saved and queued. Returns 202 with one result per file: `{name, doc_id, state}` or `{name, error}`. Needs `python-multipart`. |
| `GET /api/rag/documents` | The manifest. |
| `DELETE /api/rag/documents/{doc_id}` | Queues removal of the file and its chunks. 404 if unknown. |
| `POST /api/rag/reindex` | Queues a full re-index with the current embedder and chunk settings. |
| `GET /api/rag/embedders` | Available embedder slugs, with `available` and a reason when unavailable. |

### 4.5 WebSocket

| Direction | Message |
|---|---|
| client to server | `{"type": "update_rag", "enabled"?, "top_k"?, "min_score"?, "max_context_chars"?, "chunk_size"?, "chunk_overlap"?, "embedder"?}`. The values are validated like `update_settings`. A changed index setting, or a switch to an embedder whose collection is stale, queues a re-index. |
| server to client | `{"type": "rag_config", "settings": {...}, "embedders": [...], "collection": {"embedder", "dim", "state", "chunk_size", "chunk_overlap"}}`, sent on connect and after every change |
| server to client | `{"type": "rag_status", "documents": [...], "indexing": {"done", "total"} \| null}` |
| server to client | `{"type": "rag_sources", ...}` (§4.3) |

Switching the embedder loads it first (Hugging Face takes about 2 s; for Ollama, one test embedding). If loading fails, the previous embedder stays active and `rag_config` carries the error. The switch is refused while a turn runs.

### 4.6 Configuration (`.env`)

| Variable | Default |
|---|---|
| `RAG_DIR` | `data/rag` |
| `RAG_EMBEDDER` | `hf:BAAI/bge-small-en-v1.5` (initial value; the UI choice is then saved in `settings.json`) |
| `RAG_EMBED_DEVICE` | `cuda` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `RAG_MAX_UPLOAD_MB` | `25` |

## 5. Ollama on the VM

`scripts/ollama.sh` (LF line endings):

- `scripts/ollama.sh start` runs `docker run -d --gpus all --name voicebot-ollama -p 127.0.0.1:11434:11434 -v $REPO/models/ollama:/root/.ollama ollama/ollama`, or starts the existing container.
- `scripts/ollama.sh pull <model>` runs `ollama pull` in the container (default `nomic-embed-text`, about 270 MB).
- `scripts/ollama.sh stop` stops the container.

The server does not start Ollama itself. It only reports whether Ollama is reachable.

## 6. Dependencies

Added to the server venv: `chromadb`, `sentence-transformers`, `pypdf`, `python-docx`, `python-multipart`.

**Resolution risk.** `chromadb` depends on `onnxruntime`, and the server venv already holds faster-whisper's and sherpa-onnx's ONNX Runtime pins. The first plan task is a `uv pip install --dry-run` on the VM, plus `import sherpa_onnx, faster_whisper, chromadb` after installing. If they conflict, stop and ask the user before working around it. `sentence-transformers` must keep `torch==2.7.1+cu128`; it is installed with the existing torch pin.

## 7. Error handling

| Case | Behaviour |
|---|---|
| Unsupported type or file over the size limit | Rejected per file in the upload response; other files in the same upload continue |
| PDF without a text layer, or an empty file | State `no text`, kept in the list so the user sees it |
| Parser error | State `failed: <reason>`; the original file is kept so re-index can retry |
| Ollama unreachable at query time | `rag_error`; the turn answers without context |
| Ollama unreachable at switch time | Switch refused; previous embedder stays |
| Server restart during indexing | On start, documents left `queued` or `indexing` are queued again |
| Chroma collection settings differ from the saved settings | Collection marked "needs re-index"; retrieval keeps using it until re-indexed |

## 8. Testing

All tests run without a GPU, Docker, network or downloaded models (v1 rule).

- **Loaders:** `.txt`, `.md` and `.docx` fixtures are created in each test (`python-docx` writes the `.docx`). A one-page text PDF and an image-only PDF are committed under `server/tests/fixtures/`. Also unsupported types and empty files.
- **Chunker:** size limit, overlap, preferred split points, no empty chunks, text shorter than one chunk.
- **Embedders:** `OllamaEmbedder` against a fake HTTP transport (request shape, normalisation, timeout raises). `HFEmbedder` with a fake `SentenceTransformer` class (query prefix only on queries).
- **Store and service:** a real Chroma `PersistentClient` in `tmp_path` with a deterministic fake embedder. Covers index, delete, re-index on a chunk-setting change, one collection per embedder, the similarity threshold, `top_k`, the context character cap and restart recovery.
- **Session:** `retrieval_ms` recorded, context reaches the system message but not history, `rag_sources` sent, and a failing retriever still produces a normal reply plus `rag_error`.
- **Web API:** upload of several files with one bad file, list, delete, through FastAPI's test client.
- **Web UI:** reducer tests for `rag_config`, `rag_status` and `rag_sources`, sources attached to the right reply, the tab state, and template validation.

Manual checks on the VM, done by the user: upload the four file types; ask a question each document answers, with RAG on and off; switch to Ollama and repeat; compare with `latency_report --by rag_embedder,rag_enabled`.
