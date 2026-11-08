"""Structured Analysis Planner translating natural language into validated analysis plans."""

import json
import re
import uuid
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from neuroflow.agent.llm import get_chat_model
from neuroflow.agent.prompts import build_planner_system_prompt
from neuroflow.agent.validator import validate_analysis_plan
from neuroflow.data.experiment_context import get_default_experiment_context
from neuroflow.schemas.experiment import ExperimentContext
from neuroflow.schemas.planning import (
    AnalysisPlan,
    AnalysisStep,
    ClarificationNeeded,
    PlanningResult,
    ResearchRequest,
)
from neuroflow.tools.registry import ToolRegistry, get_tool_registry


class AnalysisPlanner:
    """Translates natural language research requests into structured, validated analysis plans."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        registry: ToolRegistry | None = None,
        context: ExperimentContext | None = None,
    ) -> None:
        self.llm = llm or get_chat_model()
        self.registry = registry or get_tool_registry()
        self.context = context or get_default_experiment_context()
        self.system_prompt = build_planner_system_prompt(self.context, self.registry)

    def plan(self, request: ResearchRequest | str) -> PlanningResult:
        """Generate a structured analysis plan or clarification from a research question.
        
        Note: Does NOT execute any analysis tools.
        """
        if isinstance(request, str):
            req = ResearchRequest(
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                query=request.strip(),
                experiment_id=self.context.experiment_id,
            )
        else:
            req = request

        # Send system prompt and query to chat model
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"Research Request ID: {req.request_id}\nResearch Question: {req.query}"
            ),
        ]

        try:
            ai_message = self.llm.invoke(messages)
            raw_text = str(ai_message.content)
        except Exception as e:
            return PlanningResult(
                status="rejected",
                validation_errors=[f"LLM generation failed: {str(e)}"],
                raw_model_response=None,
            )

        # Parse JSON from model response
        parsed_data = self._extract_json(raw_text)
        if not parsed_data:
            return PlanningResult(
                status="rejected",
                validation_errors=["Failed to parse structured JSON from LLM response."],
                raw_model_response=raw_text,
            )

        # Handle Clarification response
        if parsed_data.get("type") == "clarification" or "question" in parsed_data:
            clarification = ClarificationNeeded(
                needs_clarification=True,
                question=parsed_data.get("question", "Clarification required."),
                options=parsed_data.get("options", []),
                missing_variables=parsed_data.get("missing_variables", []),
                unsupported_features=parsed_data.get("unsupported_features", []),
                suggested_alternatives=parsed_data.get("suggested_alternatives", []),
            )
            return PlanningResult(
                status="clarification_needed",
                clarification=clarification,
                raw_model_response=raw_text,
            )

        # Handle Plan response
        try:
            steps_raw = parsed_data.get("steps", [])
            steps: list[AnalysisStep] = []
            for s in steps_raw:
                step_obj = AnalysisStep(
                    step_id=s.get("step_id", f"step_{len(steps) + 1}"),
                    tool_name=s.get("tool_name", ""),
                    purpose=s.get("purpose", ""),
                    parameters=s.get("parameters", {}),
                    rationale=s.get("rationale", ""),
                    dependencies=s.get("dependencies", []),
                )
                steps.append(step_obj)

            plan = AnalysisPlan(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                request_id=req.request_id,
                summary=parsed_data.get("summary", f"Plan for: {req.query}"),
                steps=steps,
            )

            # Validate plan against registry and experiment context
            is_valid, validation_errors = validate_analysis_plan(
                plan=plan, registry=self.registry, context=self.context
            )

            if not is_valid:
                return PlanningResult(
                    status="rejected",
                    plan=plan,
                    validation_errors=validation_errors,
                    raw_model_response=raw_text,
                )

            return PlanningResult(
                status="plan_generated",
                plan=plan,
                raw_model_response=raw_text,
            )

        except Exception as e:
            return PlanningResult(
                status="rejected",
                validation_errors=[f"Plan structure construction failed: {str(e)}"],
                raw_model_response=raw_text,
            )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract and parse JSON object from markdown code fence or raw string."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Match ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Match first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None
