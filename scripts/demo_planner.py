"""Demo script to test and inspect NeuroFlow structured analysis planning."""

import json
import sys
from pathlib import Path

# Add src to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuroflow.agent.planner import AnalysisPlanner


def main() -> None:
    print("=" * 75)
    print("NeuroFlow AI - Milestone 2 Structured Analysis Planning Demonstration")
    print("=" * 75)

    planner = AnalysisPlanner()

    sample_queries = [
        (
            "1. Valid Group Comparison Request",
            "Compare memory accuracy between the control and experimental groups.",
        ),
        (
            "2. Valid Learning Progression Request",
            "Examine learning progression over trials.",
        ),
        (
            "3. Valid Correlation Request",
            "Investigate whether response time is associated with error rate.",
        ),
        (
            "4. Ambiguous Request (Triggers Clarification)",
            "do some analysis on the participants",
        ),
        (
            "5. Unknown Variable Request (Rejected / Clarification)",
            "Compare participant blood_pressure and heart_rate across cohorts.",
        ),
        (
            "6. Unsupported Request (Rejected / Clarification)",
            "Train a deep neural network in PyTorch to classify participants.",
        ),
    ]

    for title, query in sample_queries:
        print("\n" + "-" * 75)
        print(f"CASE: {title}")
        print(f"Research Query: \"{query}\"")
        print("-" * 75)

        result = planner.plan(query)
        print(f"Planning Status: {result.status.upper()}")

        if result.status == "plan_generated" and result.plan:
            print(f"Plan Summary:    {result.plan.summary}")
            print(f"Proposed Steps ({len(result.plan.steps)}):")
            for step in result.plan.steps:
                print(f"  [{step.step_id}] Tool: {step.tool_name}")
                print(f"         Purpose:    {step.purpose}")
                print(f"         Parameters: {step.parameters}")
                print(f"         Rationale:  {step.rationale}")
                if step.dependencies:
                    print(f"         Depends On: {step.dependencies}")
        elif result.status == "clarification_needed" and result.clarification:
            print(f"Clarification:   {result.clarification.question}")
            if result.clarification.options:
                print("Options Offered:")
                for opt in result.clarification.options:
                    print(f"  * {opt}")
            if result.clarification.missing_variables:
                print(f"Missing Vars:    {result.clarification.missing_variables}")
            if result.clarification.unsupported_features:
                print(f"Unsupported:     {result.clarification.unsupported_features}")
            if result.clarification.suggested_alternatives:
                print(f"Alternatives:    {result.clarification.suggested_alternatives}")
        elif result.status == "rejected":
            print(f"Validation Errors: {result.validation_errors}")

    print("\n" + "=" * 75)
    print("All Milestone 2 planner demonstration cases executed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
