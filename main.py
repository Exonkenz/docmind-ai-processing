from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pdfminer.high_level import extract_text
import anthropic
import json
import os
import datetime

app = FastAPI(title="DocMind Processing Service")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class ProcessRequest(BaseModel):
    document_id: str
    file_contents: str
    filename: str

@app.get("/health")
async def health():
    return {"status": "ok"}

def extract_text_from_file(file_contents: str, filename: str) -> str:
    import base64
    import io
    raw = base64.b64decode(file_contents)
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        from pdfminer.high_level import extract_text
        return extract_text(io.BytesIO(raw))
    else:
        return raw.decode('utf-8')

@app.post("/process")
async def process_document(req: ProcessRequest):
    # Step 1: Extract text
    try:
        text = extract_text_from_file(req.file_contents, req.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    # Trim to 4000 chars to stay well within token limits
    text = text[:4000]

    # Step 2: Call Claude for structured extraction
    prompt = f"""Analyze this technical document and return a JSON object.

Return ONLY valid JSON with NO extra text before or after. Use this exact structure:
{{
  "workflows": [
    {{"index": 0, "description": "step description", "actor": "who"}}
  ],
  "dependencies": [
    {{"from": "A", "to": "B", "type": "hard", "description": "why"}}
  ],
  "constraints": [
    {{"text": "constraint", "polarity": "must", "subject": "subject"}}
  ],
  "open_questions": ["question 1"],
  "findings": [
    {{"type": "version_constraint", "severity": "medium", "excerpt": "text", "recommendation": "action"}}
  ]
}}

Keep each array to a maximum of 5 items. Keep all strings under 100 characters.

Document:
{text}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean up response if wrapped in markdown
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response_text = part
                    break

        result = json.loads(response_text)
        result["document_id"] = req.document_id
        result["extracted_at"] = datetime.datetime.utcnow().isoformat()
        
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM extraction failed: {str(e)}")

class SummarizeRequest(BaseModel):
    document_id: str
    extraction_result: dict
    level: str = "technical"

@app.post("/summarize")
async def summarize_document(req: SummarizeRequest):
    extraction = req.extraction_result
    level = req.level

    level_instructions = {
        "general": "Use simple, plain language. Avoid technical jargon. Write for a non-technical audience.",
        "technical": "Use precise technical language. Include specific details about systems and components.",
        "executive": "Focus on business impact, risks, and recommendations. Be concise and action-oriented."
    }

    instruction = level_instructions.get(level, level_instructions["technical"])

    workflows = extraction.get("workflows", [])
    dependencies = extraction.get("dependencies", [])
    constraints = extraction.get("constraints", [])
    findings = extraction.get("findings", [])

    context = f"""
Workflows: {json.dumps(workflows[:5])}
Dependencies: {json.dumps(dependencies[:5])}
Constraints: {json.dumps(constraints[:5])}
Findings: {json.dumps(findings[:5])}
"""

    prompt = f"""Based on this document analysis, write three summaries.

{instruction}

Document analysis:
{context}

Return ONLY valid JSON with exactly these keys:
{{
  "tl_dr": "maximum 50 words summary here",
  "executive": "maximum 200 words summary here",
  "technical": "maximum 500 words summary here"
}}

Keep strictly within word limits. Return only the JSON object."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response_text = part
                    break

        result = json.loads(response_text)
        result["document_id"] = req.document_id
        result["level"] = level

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse summary response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")