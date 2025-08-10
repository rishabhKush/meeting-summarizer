# netlify/functions/summarize.py
import json
import re
from typing import List

import spacy
import pytextrank

# Load nlp once per function cold-start
nlp = spacy.load("en_core_web_sm")
# add pytextrank pipeline (idempotent)
if "textrank" not in nlp.pipe_names:
    nlp.add_pipe("textrank")

# chunking helpers
def chunk_text(text: str, max_chars: int = 4500, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks by characters.
    Keeps sentence boundaries best-effort by splitting on newline or punctuation.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        # try to extend end to nearest sentence boundary
        if end < L:
            m = re.search(r"[\.\n]\s", text[end-50:end+50])
            if m:
                # place end at match end
                rel = m.end() - 50
                end = max(start, end - 50 + rel)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks


def summarize_chunk(chunk: str, sent_limit: int = 6) -> List[str]:
    doc = nlp(chunk)
    # pytextrank offers a summary helper
    try:
        sents = list(doc._.textrank.summary(limit_phrases=15, limit_sentences=sent_limit))
        # returns list of (sentence, score) tuples — take sentence text
        return [s[0] for s in sents]
    except Exception:
        # fallback: top sentences by length
        return [sent.text.strip() for sent in list(doc.sents)[:sent_limit]]


def extract_action_items(text: str) -> List[dict]:
    doc = nlp(text)
    actions = []
    action_keywords = ["action", "action item", "will", "shall", "to do", "assign", "assigned to", "owner", "deadline", "by "]
    for sent in doc.sents:
        s = sent.text.strip()
        low = s.lower()
        if any(k in low for k in action_keywords):
            # find possible owners and dates
            owners = [ent.text for ent in sent.ents if ent.label_ in ("PERSON", "ORG")]
            dates = [ent.text for ent in sent.ents if ent.label_ in ("DATE", "TIME")]
            actions.append({
                "text": s,
                "owners": owners,
                "dates": dates,
            })
    # dedupe
    seen = set()
    out = []
    for a in actions:
        key = a["text"]
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def extract_requirements_from_phrases(text: str, topn: int = 12) -> List[str]:
    doc = nlp(text)
    phrases = [p.text for p in doc._.phrases[:topn]]
    # filter for noun/verb containing phrases and those with keywords
    req_keywords = ["require", "need", "should", "must", "deliver", "support", "implement", "setup", "integration"]
    reqs = []
    for ph in phrases:
        low = ph.lower()
        if any(k in low for k in req_keywords) or len(ph.split()) <= 5:
            reqs.append(ph)
    # fallback: any phrase topn
    if not reqs:
        reqs = phrases[:min(len(phrases), topn)]
    return reqs


def extract_clients(text: str, topn: int = 8) -> List[str]:
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON")]
    # frequency order
    freq = {}
    for e in ents:
        freq[e] = freq.get(e, 0) + 1
    sorted_ents = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [e[0] for e in sorted_ents[:topn]]


# main handler for Netlify function

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        transcript = body.get("transcript", "")
        if not transcript:
            return {"statusCode": 400, "body": json.dumps({"error": "No transcript provided"})}

        # chunk the transcript
        chunks = chunk_text(transcript)

        # summarize each chunk and collect
        chunk_summaries = []
        for c in chunks:
            sents = summarize_chunk(c, sent_limit=5)
            chunk_summaries.extend(sents)

        # finalize summary by summarizing combined top sentences
        combined = "\n".join(chunk_summaries)
        final_summary_sents = summarize_chunk(combined, sent_limit=6)
        final_summary = " ".join(final_summary_sents)

        # action items: scan whole transcript
        action_items = extract_action_items(transcript)

        # clients (top entities)
        clients = extract_clients(transcript)

        # requirements
        requirements = extract_requirements_from_phrases(transcript)

        response = {
            "summary": final_summary,
            "action_items": action_items,
            "clients": clients,
            "requirements": requirements,
            "minutes": chunk_summaries,
        }

        return {"statusCode": 200, "body": json.dumps(response)}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
