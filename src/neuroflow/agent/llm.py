"""Configurable LLM provider integration for NeuroFlow AI."""

import json
import re
from typing import Any, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from neuroflow.config import Settings, get_settings


class FakePlannerChatModel(BaseChatModel):
    """Deterministic fake chat model for unit tests and local mock development.
    
    Generates structured JSON strings representing valid AnalysisPlan or
    ClarificationNeeded outputs according to the research query.
    """

    model_name: str = "fake-planner-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Extract the prompt from HumanMessage only (not SystemMessage)
        user_texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        prompt_text = "\n".join(user_texts) if user_texts else str(messages[-1].content)

        response_dict = self._plan_for_prompt(prompt_text)
        content_str = json.dumps(response_dict)
        generation = ChatGeneration(message=AIMessage(content=content_str))
        return ChatResult(generations=[generation])

    def _plan_for_prompt(self, text: str) -> dict[str, Any]:
        lower = text.lower()

        # 1. Check for unsupported capabilities / arbitrary scripts FIRST
        unsupported_keywords = [
            "neural network",
            "pytorch",
            "tensorflow",
            "custom python",
            "arbitrary",
            "execute sql",
            "diagnose",
            "deep learning",
        ]
        found_unsupported = [u for u in unsupported_keywords if u in lower]
        if found_unsupported:
            return {
                "type": "clarification",
                "question": "The requested analysis involves arbitrary code execution, clinical diagnosis, or unapproved ML algorithms outside the registered catalogue.",
                "options": [
                    "Use registered 'predict_task_performance' (supervised mock classification)",
                    "Use registered 'cluster_participant_behaviour' (behavioural clustering)",
                ],
                "unsupported_features": found_unsupported,
                "suggested_alternatives": ["predict_task_performance", "cluster_participant_behaviour"],
            }

        # 2. Check for unknown variable requests
        unknown_var_patterns = [
            r"\bblood_pressure\b",
            r"\bblood pressure\b",
            r"\beeg\b",
            r"\bheart_rate\b",
            r"\bheart rate\b",
            r"\bgalvanic\b",
            r"\bglucose\b",
            r"\breaction_score\b",
        ]
        found_unknown = []
        for pat in unknown_var_patterns:
            m = re.search(pat, lower)
            if m:
                found_unknown.append(m.group(0))

        if found_unknown:
            return {
                "type": "clarification",
                "question": f"The requested variable(s) '{', '.join(found_unknown)}' are not found in the experimental dataset.",
                "options": [
                    "Use 'response_time_ms' (Response latency in ms)",
                    "Use 'response_correct' (Accuracy indicator)",
                    "Use 'attempt_count' (Number of attempts)",
                ],
                "missing_variables": found_unknown,
                "suggested_alternatives": ["response_time_ms", "response_correct", "attempt_count"],
            }

        # 3. Check for ambiguous requests
        # If wrapped in "Research Question: ...", isolate the query portion
        if "research question:" in lower:
            q_part = lower.split("research question:", 1)[1].strip()
        else:
            q_part = lower.strip()

        ambiguous_triggers = [
            "analyze participants",
            "analyze the participants",
            "analyze experiment",
            "analyze my experiment",
            "do some analysis",
            "tell me about the data",
            "tell me about data",
            "inspect data",
        ]
        is_ambiguous = any(trigger in q_part for trigger in ambiguous_triggers) and len(q_part.split()) < 12
        if is_ambiguous:
            return {
                "type": "clarification",
                "question": "Your research question is broad. Which specific analytical objective would you like to investigate?",
                "options": [
                    "Compare memory performance between experimental groups",
                    "Analyze learning curves across trials",
                    "Evaluate response time latency and variability",
                    "Check data quality and flag anomalous sessions",
                ],
                "suggested_alternatives": [
                    "compare_experimental_groups",
                    "analyze_learning_curve",
                    "analyze_response_time",
                    "validate_experiment_data",
                ],
            }

        # 4. Standard deterministic plan generation
        steps: list[dict[str, Any]] = [
            {
                "step_id": "step_1",
                "tool_name": "validate_experiment_data",
                "purpose": "Audit dataset completeness, timestamps, and missing records before statistical analysis.",
                "parameters": {"task_type": "all"},
                "rationale": "Ensures underlying data integrity before running downstream analytical evaluations.",
                "dependencies": [],
            }
        ]

        step_counter = 2

        # Learning curve
        if "learning" in lower or "trials" in lower or "progression" in lower or "curve" in lower:
            metric = "accuracy" if "accuracy" in lower else ("response_time_ms" if "response time" in lower else "accuracy")
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "analyze_learning_curve",
                    "purpose": f"Evaluate {metric} progression over sequential trials.",
                    "parameters": {"metric": metric, "trial_start": 1, "trial_end": 10},
                    "rationale": "Determines whether practice effects lead to measurable cognitive improvement.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # Group comparison
        if "compare" in lower or "group" in lower or "groups" in lower or "cohort" in lower or "intervention" in lower:
            # First descriptive performance
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "analyze_memory_performance",
                    "purpose": "Establish baseline descriptive performance metrics across the cohort.",
                    "parameters": {"metric": "accuracy", "task_type": "all"},
                    "rationale": "Provides reference distribution parameters for cognitive performance metrics.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "compare_experimental_groups",
                    "purpose": "Statistically compare accuracy outcome between control and experimental cohorts.",
                    "parameters": {
                        "metric": "accuracy",
                        "group_a": "control",
                        "group_b": "experimental",
                        "task_type": "all",
                    },
                    "rationale": "Tests the primary hypothesis that the experimental intervention enhances memory performance.",
                    "dependencies": [f"step_{step_counter - 1}"],
                }
            )
            step_counter += 1

        # Correlations
        if "correlation" in lower or "associated" in lower or "relationship" in lower or "association" in lower:
            x_var = "response_time_ms"
            y_var = "error_rate"
            if "accuracy" in lower and "response time" in lower:
                y_var = "accuracy"
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "analyze_performance_correlations",
                    "purpose": f"Quantify the association between '{x_var}' and '{y_var}'.",
                    "parameters": {
                        "x_variable": x_var,
                        "y_variable": y_var,
                        "correlation_method": "pearson",
                    },
                    "rationale": "Examines potential speed-accuracy trade-offs or cognitive processing dynamics.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # Task difficulty
        if "difficulty" in lower or "lowest accuracy" in lower or "hardest" in lower or "demand" in lower:
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "analyze_task_difficulty",
                    "purpose": "Rank and compare relative difficulty across memory tasks.",
                    "parameters": {"metric": "accuracy", "sort_order": "ascending"},
                    "rationale": "Identifies task types with higher cognitive load or error susceptibility.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # Anomalies
        if "anomal" in lower or "unusual" in lower or "incomplete" in lower or "flagged" in lower:
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "detect_anomalous_sessions",
                    "purpose": "Detect session runs with atypical timing or error bursts.",
                    "parameters": {"sensitivity": "medium"},
                    "rationale": "Flags outlier participant runs for closer scientific inspection.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # Clustering
        if "cluster" in lower or "phenotype" in lower or "behavioural" in lower or "behavioral" in lower:
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "cluster_participant_behaviour",
                    "purpose": "Partition participants into distinct behavioural strategy profiles.",
                    "parameters": {
                        "n_clusters": 3,
                        "features": ["response_time_ms", "error_rate", "accuracy"],
                    },
                    "rationale": "Identifies underlying behavioural phenotypes across participants.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # Prediction
        if "predict" in lower or "classification" in lower or "supervised" in lower:
            steps.append(
                {
                    "step_id": f"step_{step_counter}",
                    "tool_name": "predict_task_performance",
                    "purpose": "Evaluate predictive power of behavioural features on high accuracy.",
                    "parameters": {
                        "target": "high_accuracy",
                        "features": ["response_time_ms", "attempt_count", "trial_number"],
                    },
                    "rationale": "Determines which cognitive performance features best predict successful trial outcomes.",
                    "dependencies": ["step_1"],
                }
            )
            step_counter += 1

        # If nothing specific was triggered, provide baseline memory performance analysis
        if len(steps) == 1:
            steps.append(
                {
                    "step_id": "step_2",
                    "tool_name": "analyze_memory_performance",
                    "purpose": "Provide descriptive summary of participant accuracy and response times.",
                    "parameters": {"metric": "accuracy", "task_type": "all"},
                    "rationale": "Establishes baseline experimental performance metrics.",
                    "dependencies": ["step_1"],
                }
            )

        summary = f"Structured analytical workflow addressing research question: '{text.strip()}'"

        return {
            "type": "plan",
            "summary": summary,
            "steps": steps,
        }

    @property
    def _llm_type(self) -> str:
        return "fake-planner"


def get_chat_model(cfg: Settings | None = None) -> BaseChatModel:
    """Factory creating the configured LangChain chat model."""
    config = cfg or get_settings()

    if config.llm_provider == "fake":
        return FakePlannerChatModel(model_name=config.llm_model)

    if config.llm_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.llm_model,
                base_url=config.ollama_base_url,
                temperature=config.llm_temperature,
            )
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=config.llm_model,
                    base_url=config.ollama_base_url,
                    temperature=config.llm_temperature,
                )
            except ImportError as e:
                raise ImportError(
                    "Ollama provider requires 'langchain_ollama' or 'langchain_community'. "
                    "Install with 'pip install langchain-ollama'."
                ) from e

    if config.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                api_key=config.openai_api_key,
            )
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires 'langchain_openai'. Install with 'pip install langchain-openai'."
            ) from e

    if config.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.llm_model,
                temperature=config.llm_temperature,
                api_key=config.anthropic_api_key,
            )
        except ImportError as e:
            raise ImportError(
                "Anthropic provider requires 'langchain_anthropic'. Install with 'pip install langchain-anthropic'."
            ) from e

    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
