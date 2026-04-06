"""
chunker.py — Load .txt files and split them into overlapping text chunks.

WHY CHUNK?
  Language models and embedding models have token limits. We split large documents
  into smaller pieces ("chunks") so each piece can be independently embedded and
  stored in a vector database. Overlap ensures we don't lose context at boundaries.

WHAT THIS PRODUCES:
  A list of dicts, each with:
    - id:        unique identifier  (used as Cosmos DB document id)
    - source:    original filename
    - section:   heading of the section the chunk came from (if any)
    - chunk_idx: sequential chunk number within the file
    - text:      the actual chunk text
"""

import os
import re
import uuid
import hashlib


# ── Tunables ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800       # target characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks
TXT_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script


def _detect_section(line: str) -> str | None:
    """Return a section heading if the line looks like one, else None."""
    stripped = line.strip()
    # Matches lines like "Route Performance & On-Time Metrics" followed by dashes
    if stripped and not stripped.startswith("-"):
        return stripped
    return None


def _split_into_sections(text: str) -> list[dict]:
    """
    Split document text at section boundaries (lines followed by a row of dashes).
    Returns a list of {"section": ..., "text": ...} dicts.
    """
    lines = text.split("\n")
    sections = []
    current_section = "Introduction"
    current_lines = []

    i = 0
    while i < len(lines):
        # Check if the NEXT line is a dash-underline (indicates current line is a heading)
        if (i + 1 < len(lines)
                and re.match(r'^-{3,}|^={3,}', lines[i + 1].strip())
                and lines[i].strip()):
            # Save accumulated text under previous section
            if current_lines:
                sections.append({"section": current_section,
                                 "text": "\n".join(current_lines).strip()})
                current_lines = []
            current_section = lines[i].strip()
            i += 2  # skip heading + underline
            continue
        current_lines.append(lines[i])
        i += 1

    # Don't forget the last section
    if current_lines:
        sections.append({"section": current_section,
                         "text": "\n".join(current_lines).strip()})
    return sections


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sliding-window chunker that tries to break on paragraph/sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph boundary (\n\n)
        if end < len(text):
            break_point = text.rfind("\n\n", start, end)
            if break_point == -1 or break_point <= start:
                # Fall back to sentence boundary
                break_point = text.rfind(". ", start, end)
            if break_point != -1 and break_point > start:
                end = break_point + 1  # include the period / newline

        chunks.append(text[start:end].strip())
        start = end - overlap  # slide back by overlap amount

    return [c for c in chunks if c]  # drop empties


def _make_id(source: str, chunk_idx: int) -> str:
    """Deterministic document id from filename + chunk index."""
    raw = f"{source}::{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_and_chunk(directory: str = TXT_DIR) -> list[dict]:
    """
    Main entry-point.  Reads every .txt file in `directory`, splits into
    section-aware overlapping chunks, and returns a list of chunk dicts.
    """
    chunks = []
    txt_files = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))

    for filename in txt_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, encoding="utf-8") as fh:
            content = fh.read()

        sections = _split_into_sections(content)
        chunk_idx = 0

        for sec in sections:
            for piece in _chunk_text(sec["text"]):
                chunks.append({
                    "id": _make_id(filename, chunk_idx),
                    "source": filename,
                    "section": sec["section"],
                    "chunk_idx": chunk_idx,
                    "text": piece,
                })
                chunk_idx += 1

    return chunks


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_chunks = load_and_chunk()
    print(f"Total chunks: {len(all_chunks)}\n")
    for c in all_chunks[:3]:
        print(f"[{c['source']}  §{c['section']}  #{c['chunk_idx']}]")
        print(c["text"][:200], "...\n")
