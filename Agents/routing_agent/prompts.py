"""
prompts.py
==========

Prompt templates for the OPTIONAL LLM-in-the-loop hooks. Not called by the
agent by default (config.ENABLE_LLM_SCORING is False). Wire a real model by
implementing tools.CallableLLMScorer with a function that fills these
templates, calls the model, and parses the strict-JSON response.

Design constraint enforced by the templates themselves: the model must
return ONLY a final structured judgement (decision / evidence / reason /
confidence-relevant score) -- never its intermediate chain-of-thought. This
matches the "Explainable Output" requirement: hidden reasoning is never
exposed, only distilled evidence.
"""

LLM_ROUTING_EVALUATION_PROMPT = """You are assisting a document routing system.

Given the DOCUMENT TEXT and a candidate DEPARTMENT description, judge how
well this department fits as the correct recipient of the document.

DOCUMENT TEXT:
{document_text}

CANDIDATE DEPARTMENT:
Name: {department_name}
Responsibilities: {responsibilities}
Handled topics: {handled_topics}
Excluded topics: {excluded_topics}

Respond with ONLY a single JSON object, no other text, no explanation of
your reasoning process:

{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}
"""

INTENT_EXTRACTION_PROMPT = """Identify the distinct requests/intents contained in this document.
A document has multiple intents only if it clearly asks for more than one
unrelated action from different departments.

DOCUMENT TEXT:
{document_text}

Respond with ONLY a JSON array, no other text:

[{{"label": "<short label>", "text": "<the portion of text for this intent>"}}]
"""

CONFLICT_CHECK_PROMPT = """Compare these three upstream signals about the same document and say
whether they are consistent with routing it to the given department.

Classification topic: {classification_topic}
Analysis summary: {analysis_summary}
Writing output type: {writing_type}
Proposed routing department: {department_name}

Respond with ONLY a JSON object, no other text:

{{"consistent": <true/false>, "reason": "<one short sentence>"}}
"""
