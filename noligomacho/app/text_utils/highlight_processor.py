def extract_highlight_with_tail(doc, field="text", tail_length=1000):
    full_text = doc.get('_source', {}).get(field, "")

    highlight_fragments = doc.get('highlight', {}).get(field, [])
    highlight_text = " [...] ".join(highlight_fragments)

    tail_text = full_text[-tail_length:] if full_text else ""

    return f"{highlight_text} [...] {tail_text}".strip()
