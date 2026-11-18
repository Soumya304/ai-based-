"""Run: python -m pytest tests"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ingest import chunk, CHUNK_WORDS


def test_chunk_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(600))
    parts = list(chunk(text))
    assert all(len(p.split()) <= CHUNK_WORDS for p in parts)
    assert parts[0].split()[-1] != parts[1].split()[-1]
    assert set(parts[0].split()) & set(parts[1].split())  # overlap exists


def test_short_text_dropped():
    assert list(chunk("too short")) == []
