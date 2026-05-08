
# Optional provider integration sketch

def openai_summarize_report(client, text: str) -> str:
    prompt = f"""Summarize this intelligence-style report in 2 sentences.
Report:
{text}
"""
    raise NotImplementedError("Wire your preferred API client here.")
