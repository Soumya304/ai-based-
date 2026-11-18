"""Step 1: Build the vector index from study material in data/<exam>/<subject>/*.pdf|txt"""
import json
from pathlib import Path
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA, INDEX = Path("data"), Path("index")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # English + Hindi
CHUNK_WORDS, OVERLAP = 250, 50


def read_file(p: Path):
    """Return list of (page_no, text)."""
    if p.suffix.lower() == ".pdf":
        return [(i + 1, pg.extract_text() or "") for i, pg in enumerate(PdfReader(str(p)).pages)]
    return [(1, p.read_text(encoding="utf-8", errors="ignore"))]


def chunk(text: str):
    words, step = text.split(), CHUNK_WORDS - OVERLAP
    for i in range(0, len(words), step):
        part = words[i:i + CHUNK_WORDS]
        if len(part) >= 30:
            yield " ".join(part)


def main():
    meta, texts = [], []
    for f in sorted(DATA.rglob("*")):
        if f.suffix.lower() not in {".pdf", ".txt"}:
            continue
        parts = f.relative_to(DATA).parts
        exam = parts[0]
        subject = parts[1] if len(parts) > 2 else "General"
        for page, txt in read_file(f):
            for c in chunk(txt):
                meta.append({"exam": exam, "subject": subject, "file": f.name, "page": page, "text": c})
                texts.append(c)
    if not texts:
        raise SystemExit("No documents found in data/<exam>/<subject>/")
    model = SentenceTransformer(MODEL)
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity (vectors normalised)
    index.add(vecs)
    INDEX.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX / "faiss.index"))
    (INDEX / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(texts)} chunks from {len({m['file'] for m in meta})} files")


if __name__ == "__main__":
    main()
