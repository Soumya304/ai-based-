"""Flask backend: serves web/index.html and exposes the RAG pipeline as a JSON API.
Run: python server.py  ->  http://localhost:5000
"""
from flask import Flask, jsonify, request, send_from_directory
from rag import Retriever, answer, make_quiz, study_plan

app = Flask(__name__, static_folder="web", static_url_path="")
_ret = None


def retriever():
    global _ret
    if _ret is None:
        _ret = Retriever()
    return _ret


@app.errorhandler(Exception)
def fail(e):
    return jsonify(error=str(e)), 500


@app.get("/")
def home():
    return send_from_directory("web", "index.html")


@app.get("/api/meta")
def meta():
    r = retriever()
    return jsonify(exams=r.exams(), subjects={e: r.subjects(e) for e in r.exams()})


@app.post("/api/ask")
def ask():
    d = request.get_json(force=True)
    ans, hits = answer(retriever(), d["q"], d.get("exam") or None, d.get("subject") or None)
    return jsonify(answer=ans, sources=[{k: h[k] for k in ("file", "subject", "page", "score", "text")} for h in hits])


@app.post("/api/quiz")
def quiz():
    d = request.get_json(force=True)
    topic = d.get("topic") or "important facts"
    qs = make_quiz(retriever(), topic, int(d.get("n", 5)), d.get("exam") or None, d.get("subject") or None)
    return jsonify(questions=qs)


@app.post("/api/plan")
def plan():
    d = request.get_json(force=True)
    return jsonify(plan=study_plan(d.get("exam", "UPSC"), d.get("days", 90), d.get("hours", 6), d.get("weak", "")))


if __name__ == "__main__":
    app.run(debug=True)
