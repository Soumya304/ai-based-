# AI-Based Government Exam Preparation Assistant (RAG)

## 1. Abstract
Aspirants of UPSC, SSC, Banking, Railways and State PSC exams study from many PDFs and notes, and general chatbots often give unverified or hallucinated answers. This project builds a Retrieval-Augmented Generation (RAG) assistant that answers questions strictly from the aspirant's own trusted study material, cites the source page, generates practice MCQs, and creates personalised study plans.

## 2. Objectives
- Answer exam questions grounded in verified material, with citations
- Reduce hallucination by refusing when context is missing
- Generate topic-wise MCQs with explanations
- Support English and Hindi queries
- Produce personalised study plans

## 3. Architecture
```
Study PDFs/TXT --> Chunking (250 words, 50 overlap) --> Embeddings (multilingual MiniLM)
                                                              |
                                                        FAISS vector index
                                                              |
User query --> Embed --> Top-k search (filter exam/subject) --> Prompt + context --> LLM (Claude) --> Answer + citations
```

## 4. Tech stack
Python, Flask (API), Streamlit (alternate UI), sentence-transformers (embeddings), FAISS (vector search), pypdf (parsing), Anthropic API (generation), HTML/CSS/JavaScript (web client).

## 5. Project structure
| File | Role |
|---|---|
| ingest.py | Parse PDF/TXT, chunk, embed, build FAISS index with exam/subject/page metadata |
| rag.py | Retriever, grounded Q&A, quiz generator, study planner |
| server.py | Flask backend: serves the web client and the /api/ask, /api/quiz, /api/plan, /api/meta endpoints |
| web/index.html | Web client. Works alone (in-browser BM25 mode) or with the backend (server mode, detected automatically) |
| app.py | Optional Streamlit UI using the same rag.py |
| evaluate.py | Hit@k and MRR retrieval metrics (uses eval.jsonl) |
| tests/test_ingest.py | Unit tests for chunking |

## 6. Setup and run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
# put material as data/<Exam>/<Subject>/file.pdf  (e.g. data/UPSC/Polity/laxmikanth.pdf)
python ingest.py            # build the index
python server.py            # open http://localhost:5000  (HTML client + Python RAG)
# optional: streamlit run app.py     |  python evaluate.py     |  python -m pytest tests
```
To try the client without Python, open web/index.html directly: it runs in the browser with its own keyword search over notes you paste or upload.
Use only study material you have the right to use (NCERTs, government publications, your own notes).

## 7. Key design decisions
- **Page-level chunking with overlap** keeps page numbers for citations.
- **Metadata filters** (exam, subject) improve precision.
- **Similarity threshold (0.25)** triggers "not found" instead of guessing.
- **Grounded system prompt** forces answers from context only.
- **Multilingual embeddings** allow Hindi/English queries.

## 8. Evaluation
Create `eval.jsonl` (question + expected source file), then run `python evaluate.py`. Report Hit@5 and MRR. For answer quality, manually score faithfulness (supported by context?) and relevance on 30-50 questions; compare against a no-retrieval baseline.

## 9. Limitations
Scanned PDFs need OCR; tables/figures parse poorly; quality depends on the material; exact-match search for article numbers can be improved with hybrid search.

## 10. Future work
Hybrid BM25 + vector search, cross-encoder reranking, OCR, daily current-affairs ingestion, previous-year-paper analysis, weak-topic tracking, voice and mobile app.

## 11. Likely viva questions
- Why RAG instead of fine-tuning? (updatable, citable, cheaper, less hallucination)
- Why cosine similarity / normalised vectors? 
- How do you choose chunk size and overlap?
- How do you measure retrieval vs generation quality?
