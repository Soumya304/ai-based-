"""Retrieval evaluation. eval.jsonl lines: {"question": "...", "file": "expected_source.pdf"}"""
import json
from rag import Retriever

K = 5
ret = Retriever()
rows = [json.loads(l) for l in open("eval.jsonl", encoding="utf-8")]
hit = mrr = 0
for r in rows:
    files = [h["file"] for h in ret.search(r["question"], K)]
    if r["file"] in files:
        hit += 1
        mrr += 1 / (files.index(r["file"]) + 1)
print(f"Hit@{K}: {hit/len(rows):.2%}   MRR: {mrr/len(rows):.3f}   (n={len(rows)})")
