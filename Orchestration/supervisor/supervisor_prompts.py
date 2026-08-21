"""
supervisor_prompts.py
----------------------
Prompt templates for a future LLM-assisted Supervisor (`supervisor.mode:
llm` in config.yaml).

The Supervisor is fully deterministic today (see supervisor_agent.py /
routing_logic.py) and does not call any LLM. These templates are kept here,
unused, so that when/if a non-deterministic supervisor decision mode is
needed (e.g. ambiguous routing that benefits from reasoning over free-text
classification/validation notes), the prompt contract is already defined
and reviewed rather than improvised inline.

Templates only reference structured state fields, never raw document
content, to avoid leaking sensitive document content into logs/LLM calls.
"""

from __future__ import annotations

NEXT_STAGE_DECISION_PROMPT = """\
You are the Orchestration Supervisor for a document-processing workflow.
Decide the next stage given the current workflow state summary below.

Workflow id: {workflow_id}
Completed stages: {history_summary}
Current stage: {current_stage}
Current stage status: {current_status}
Stage results available: {available_sections}
Errors so far: {errors_summary}

Available next stages: {candidate_stages}

Respond with a single stage name from the available next stages, or "END"
to finish the workflow, or "TERMINATE" to abort. Do not invent a stage
name that is not listed.
"""

RETRY_OR_FALLBACK_PROMPT = """\
Stage "{stage}" failed with status "{status}" after {attempts} attempt(s).
Error type: {error_type}. Error message: {error_message}

Configured fallback policy: {fallback_policy}
Configured fallback stage (if any): {fallback_stage}

Should the workflow retry, fall back, skip this stage, or terminate?
Respond with exactly one of: retry, fallback, skip, terminate.
"""

__all__ = ["NEXT_STAGE_DECISION_PROMPT", "RETRY_OR_FALLBACK_PROMPT"]
