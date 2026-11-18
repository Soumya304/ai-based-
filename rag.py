"""Step 2: Retrieval + generation logic."""
import json, os, re
import faiss
import anthropic
from sentence_transformers import SentenceTransformer
from ingest import INDEX, MODEL

LLM = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
MIN_SCORE = 0.25  # below this, treat as "not in study material"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

QA_SYSTEM = """You are a government-exam tutor (UPSC, SSC, Banking, Railways, State PSC).
Answer ONLY from the numbered context passages. Cite sources like [1], [2].
If the context is insufficient, say so clearly and do not guess.
Structure: short direct answer, then key points, then 'Exam tip' (one line).
Reply in the same language as the question."""


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX / "faiss.index"))
        self.meta = json.loads((INDEX / "meta.json").read_text(encoding="utf-8"))
        self.model = SentenceTransformer(MODEL)

    def exams(self):
        return sorted({m["exam"] for m in self.meta})

    def subjects(self, exam=None):
        return sorted({m["subject"] for m in self.meta if not exam or m["exam"] == exam})

    def search(self, query, k=5, exam=None, subject=None):
        v = self.model.encode([query], normalize_embeddings=True).astype("float32")
        pool = k * 20 if (exam or subject) else k
        scores, ids = self.index.search(v, pool)
        hits = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            m = self.meta[i]
            if (exam and m["exam"] != exam) or (subject and m["subject"] != subject):
                continue
            hits.append({**m, "score": float(s)})
            if len(hits) == k:
                break
        return hits


def _llm(system, user, max_tokens=1200):
    r = client.messages.create(model=LLM, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")


def _context(hits):
    return "\n\n".join(f"[{i+1}] ({h['file']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits))


def answer(ret, question, exam=None, subject=None, k=5):
    hits = [h for h in ret.search(question, k, exam, subject) if h["score"] >= MIN_SCORE]
    if not hits:
        return "I couldn't find this in your study material. Try rephrasing or add relevant notes.", []
    return _llm(QA_SYSTEM, f"Context:\n{_context(hits)}\n\nQuestion: {question}"), hits


def make_quiz(ret, topic, n=5, exam=None, subject=None):
    hits = [h for h in ret.search(topic, 6, exam, subject) if h["score"] >= MIN_SCORE]
    if not hits:
        return []
    prompt = f"""From the context only, write {n} exam-style MCQs on '{topic}'.
Return ONLY a JSON list: [{{"q":"","options":["A","B","C","D"],"answer":0,"explanation":""}}]
('answer' is the index of the correct option.)

Context:
{_context(hits)}"""
    raw = _llm("You write accurate government-exam MCQs. Output valid JSON only.", prompt, 2000)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def study_plan(exam, days, hours, weak):
    prompt = f"""Create a {days}-day study plan for {exam}. Daily hours: {hours}.
Weak areas: {weak or 'none stated'}. Include daily topics, revision slots,
weekly mock tests, and current-affairs time. Use plain text only (no markdown, no tables): one line per week."""
    return _llm("You are an experienced government-exam mentor.", prompt, 2000)
