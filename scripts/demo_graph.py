"""Demo script showcasing LangGraph stateful approval workflow and tool execution."""

import json
import sys
from pathlib import Path

# Add src to python path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver
from neuroflow.agent.graph import (
    create_research_graph,
    resume_research_session,
    run_research_session,
)
from neuroflow.schemas.session import ApprovalDecision


def main() -> None:
    print("=" * 75)
    print("NeuroFlow AI - Milestone 3 LangGraph Approval Workflow Demonstration")
    print("=" * 75)

    graph = create_research_graph(checkpointer=MemorySaver())
    thread_id = "demo_session_001"

    query = "Compare memory accuracy between control and experimental groups."
    print(f"\n1. Submitting Research Query: \"{query}\" (Thread ID: {thread_id})")
    print("-" * 75)

    # Execute graph until approval gate
    state = run_research_session(graph=graph, query=query, thread_id=thread_id)

    # Inspect paused graph state
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"Workflow State:   PAUSED at {snapshot.next}")
    plan = snapshot.values["plan"]
    print(f"Proposed Plan ID: {plan.plan_id}")
    print(f"Summary:          {plan.summary}")
    print(f"Proposed Steps ({len(plan.steps)}):")
    for s in plan.steps:
        print(f"  * [{s.step_id}] {s.tool_name:32} | {s.purpose}")

    print("\nVerification: At this point, NO tool has executed yet.")
    print(f"Executed results count: {len(snapshot.values.get('results', {}))}")

    print("\n" + "-" * 75)
    print("2. Researcher Explicitly Approves the Plan")
    print("-" * 75)

    decision = ApprovalDecision(
        plan_id=plan.plan_id,
        approved=True,
        feedback="Plan reviewed and approved by researcher.",
    )
    print("Sending ApprovalDecision(approved=True) to resume graph thread...")

    final_state = resume_research_session(
        graph=graph, thread_id=thread_id, decision=decision
    )

    print(f"\nResumed Workflow Status: {final_state['status'].upper()}")
    print(f"Executed Tools Count:    {len(final_state['execution_records'])}")
    for record in final_state["execution_records"]:
        print(f"  * Step {record.step_id} ({record.tool_name}): {record.status.upper()}")

    print("\n" + "-" * 75)
    print("3. Synthesized Final Report (Grounded in Tool Outputs)")
    print("-" * 75)
    report = final_state["report"]
    print(report.synthesis)
    print(f"\nMock Data Notice: {report.mock_data_notice}")

    print("\n" + "=" * 75)
    print("Milestone 3 LangGraph human approval workflow demonstrated successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
