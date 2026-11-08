"""Gradio Blocks web application for NeuroFlow AI research workspace."""

import json
import os
import uuid
from typing import Any

# Disable telemetry and external version checks to ensure fast local startup without network hangs
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr
import plotly.graph_objects as go

from neuroflow.agent.graph import (
    create_research_graph,
    resume_research_session,
    run_research_session,
)
from neuroflow.agent.validator import validate_analysis_plan
from neuroflow.config import get_settings
from neuroflow.data.experiment_context import get_default_experiment_context
from neuroflow.persistence.sessions import get_session_repository
from neuroflow.schemas.planning import AnalysisPlan, AnalysisStep
from neuroflow.schemas.session import ApprovalDecision
from neuroflow.tools.registry import get_tool_registry
from neuroflow.ui.components import (
    convert_chart_data_to_figure,
    format_clarification_markdown,
    format_execution_log,
    format_plan_markdown,
    format_report_markdown,
    format_results_markdown,
)


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks UI application."""
    context = get_default_experiment_context()
    registry = get_tool_registry()

    # Pre-compile the LangGraph graph
    graph = create_research_graph()

    with gr.Blocks(
        title="NeuroFlow AI - Research Data Analysis Platform",
        analytics_enabled=False,
    ) as demo:
        # Session state variables
        state_thread_id = gr.State(lambda: f"session_{uuid.uuid4().hex[:8]}")
        state_plan = gr.State(None)
        state_results = gr.State({})

        gr.Markdown(
            """
# 🧠 NeuroFlow AI
### Agentic Experimental Data Analysis Platform (Human-in-the-Loop)
*Interactive research data analysis assistant for memory-task experiments. Proposes structured analysis plans and executes Python tools upon explicit researcher approval.*
"""
        )

        with gr.Tabs() as tabs:
            # ==============================================================
            # TAB 1: Research Workspace
            # ==============================================================
            with gr.Tab("1. Research Workspace", id="tab_workspace"):
                with gr.Accordion("📋 Active Experiment Context & Data Dictionary", open=False):
                    var_list = ", ".join(f"`{v}`" for v in context.get_variable_names())
                    gr.Markdown(
                        f"""
**Study:** {context.title} (`{context.experiment_id}`)  
**Description:** {context.description}  
**Cohorts:** {', '.join(context.groups)} | **Tasks:** {', '.join(context.task_types)} | **N = {context.total_participants}**  
**Available Variables:** {var_list}
"""
                    )

                with gr.Row():
                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            label="Natural Language Research Question",
                            placeholder="e.g. Compare memory accuracy between control and experimental groups...",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**Quick Demo Templates:**")
                        btn_sample_1 = gr.Button("Group Comparison", size="sm")
                        btn_sample_2 = gr.Button("Learning Progression", size="sm")
                        btn_sample_3 = gr.Button("RT & Error Correlation", size="sm")
                        btn_sample_4 = gr.Button("Task Difficulty", size="sm")
                        btn_sample_5 = gr.Button("Anomaly Scan", size="sm")

                with gr.Row():
                    btn_submit = gr.Button("🚀 Submit Research Request", variant="primary")
                    btn_reset = gr.Button("🔄 Start New Session", variant="secondary")

                status_box = gr.Markdown(
                    "**Status:** `Ready for research inquiry.`",
                )

                clarification_box = gr.Markdown(visible=False)

            # ==============================================================
            # TAB 2: Analysis Plan and Approval Gate
            # ==============================================================
            with gr.Tab("2. Analysis Plan & Approval", id="tab_plan"):
                gr.Markdown(
                    """
> 🛡️ **Human Approval Gate:** Execution will NOT begin automatically.  
> Review the proposed analytical operations, edit or remove steps if desired, and provide explicit approval.
"""
                )

                plan_display = gr.Markdown(
                    "*No analysis plan generated yet. Submit a query in Tab 1.*"
                )

                with gr.Accordion("⚙️ Edit or Reorder Plan Steps (Advanced)", open=False):
                    gr.Markdown(
                        "You can edit tool parameters directly in the JSON below before approving. Modified plans will be automatically re-validated."
                    )
                    steps_json_editor = gr.Textbox(
                        label="Plan Steps (Editable JSON)",
                        lines=8,
                        max_lines=20,
                    )
                    with gr.Row():
                        step_to_remove = gr.Dropdown(
                            label="Remove Specific Step",
                            choices=[],
                            interactive=True,
                        )
                        btn_remove_step = gr.Button("🗑️ Remove Selected Step")

                with gr.Row():
                    btn_approve = gr.Button(
                        "✅ Approve and Execute Plan",
                        variant="primary",
                        size="lg",
                    )
                    with gr.Column():
                        feedback_input = gr.Textbox(
                            label="Rejection / Revision Feedback (Optional)",
                            placeholder="Explain what to change or which tools to prioritize...",
                            lines=1,
                        )
                        btn_reject = gr.Button(
                            "❌ Reject Plan & Request Revision",
                            variant="stop",
                        )

            # ==============================================================
            # TAB 3: Results & Evidence Synthesis
            # ==============================================================
            with gr.Tab("3. Results & Reports", id="tab_results"):
                mock_banner = gr.Markdown(
                    """
> ⚠️ **DEMO / MOCK DATA NOTICE:**  
> All statistics, numbers, and charts displayed below are deterministic synthetic placeholders generated for interface prototyping.
"""
                )

                with gr.Accordion("📜 Execution Audit Log", open=True):
                    execution_log_box = gr.Markdown("*No tools executed yet.*")

                gr.Markdown("### Interactive Visualisations & Tool Outputs")
                with gr.Row():
                    plot_1 = gr.Plot(label="Analysis Chart 1", visible=False)
                    plot_2 = gr.Plot(label="Analysis Chart 2", visible=False)
                with gr.Row():
                    plot_3 = gr.Plot(label="Analysis Chart 3", visible=False)
                    plot_4 = gr.Plot(label="Analysis Chart 4", visible=False)

                results_details_box = gr.Markdown("*Execution outputs pending.*")

                report_box = gr.Markdown("### Synthesis Report\n*Pending execution.*")

                btn_export = gr.Button("💾 Export Full Synthesis Report")
                export_file = gr.File(label="Downloaded Report", visible=False)

            # ==============================================================
            # TAB 4: Session History & Audit
            # ==============================================================
            with gr.Tab("4. Session History", id="tab_history"):
                gr.Markdown("### 📜 Stored Research Sessions (SQLite Audit History)")
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Past Sessions",
                        placeholder="Search by query keyword or session ID...",
                        lines=1,
                    )
                    btn_search_sessions = gr.Button("🔍 Search")
                    btn_refresh_sessions = gr.Button("🔄 Refresh List")

                sessions_table = gr.Dataframe(
                    headers=["Session ID", "Created At", "Status", "Research Query", "Tools Run"],
                    datatype=["str", "str", "str", "str", "number"],
                    interactive=False,
                )

                with gr.Row():
                    selected_session_dropdown = gr.Dropdown(
                        label="Select Session to Reopen & Inspect",
                        choices=[],
                        interactive=True,
                    )
                    btn_inspect_session = gr.Button("📂 Reopen & Inspect Session", variant="primary")

                session_details_markdown = gr.Markdown(
                    "*Select a session above to inspect its full execution history, tools run, and report.*"
                )

        # ==============================================================
        # EVENT HANDLERS
        # ==============================================================

        # Quick demo templates click handlers
        btn_sample_1.click(
            lambda: "Compare memory accuracy between the control and experimental groups.",
            inputs=None,
            outputs=query_input,
        )
        btn_sample_2.click(
            lambda: "Examine learning progression over trials 1 to 10.",
            inputs=None,
            outputs=query_input,
        )
        btn_sample_3.click(
            lambda: "Investigate whether response time is associated with error rate.",
            inputs=None,
            outputs=query_input,
        )
        btn_sample_4.click(
            lambda: "Identify which memory tasks have the lowest accuracy and highest difficulty.",
            inputs=None,
            outputs=query_input,
        )
        btn_sample_5.click(
            lambda: "Find unusual or anomalous sessions with high latency.",
            inputs=None,
            outputs=query_input,
        )

        def handle_submit(query: str, thread_id: str):
            """Submit query to LangGraph workflow up to the approval gate."""
            if not query or not query.strip():
                return (
                    "**Status:** ⚠️ *Please enter a research question first.*",
                    gr.update(visible=False),
                    "*No plan generated.*",
                    "",
                    gr.update(choices=[]),
                    None,
                    gr.update(selected="tab_workspace"),
                )

            # Invoke graph thread
            res_state = run_research_session(graph=graph, query=query.strip(), thread_id=thread_id)
            status = res_state.get("status")

            # Check if clarification was needed
            if status == "clarification_needed" and res_state.get("clarification"):
                clar_md = format_clarification_markdown(res_state["clarification"])
                return (
                    "**Status:** ⚠️ *Clarification is required before a plan can be established.*",
                    gr.update(value=clar_md, visible=True),
                    "*Plan generation paused pending researcher clarification.*",
                    "",
                    gr.update(choices=[]),
                    None,
                    gr.update(selected="tab_workspace"),
                )

            # Check if plan was generated and paused for approval
            plan: AnalysisPlan | None = res_state.get("plan")
            if plan:
                plan_md = format_plan_markdown(plan)
                steps_data = [s.model_dump() for s in plan.steps]
                steps_json = json.dumps(steps_data, indent=2)
                step_ids = [s.step_id for s in plan.steps]

                status_msg = (
                    f"**Status:** 🛑 **PAUSED AT APPROVAL GATE.** Proposed plan `{plan.plan_id}` "
                    f"has been validated. Review in Tab 2 and approve to execute."
                )

                return (
                    status_msg,
                    gr.update(visible=False),
                    plan_md,
                    steps_json,
                    gr.update(choices=step_ids, value=step_ids[0] if step_ids else None),
                    plan,
                    gr.update(selected="tab_plan"),
                )

            return (
                f"**Status:** ❌ *Plan generation rejected:* {res_state.get('validation_errors', ['Unknown error'])}",
                gr.update(visible=False),
                "*Plan rejected.*",
                "",
                gr.update(choices=[]),
                None,
                gr.update(selected="tab_workspace"),
            )

        btn_submit.click(
            handle_submit,
            inputs=[query_input, state_thread_id],
            outputs=[
                status_box,
                clarification_box,
                plan_display,
                steps_json_editor,
                step_to_remove,
                state_plan,
                tabs,
            ],
        )

        def handle_remove_step(step_id: str, steps_json_str: str, current_plan: AnalysisPlan | None):
            """Remove a step from the plan and update JSON and markdown."""
            if not current_plan or not step_id:
                return steps_json_str, format_plan_markdown(current_plan), current_plan, gr.update()

            try:
                data = json.loads(steps_json_str) if steps_json_str else [s.model_dump() for s in current_plan.steps]
                filtered = [s for s in data if s.get("step_id") != step_id]

                updated_steps = [AnalysisStep(**s) for s in filtered]
                updated_plan = current_plan.model_copy(deep=True)
                updated_plan.steps = updated_steps

                new_json = json.dumps(filtered, indent=2)
                new_md = format_plan_markdown(updated_plan)
                remaining_ids = [s.step_id for s in updated_steps]

                return (
                    new_json,
                    new_md,
                    updated_plan,
                    gr.update(choices=remaining_ids, value=remaining_ids[0] if remaining_ids else None),
                )
            except Exception as e:
                return steps_json_str, f"Error removing step: {e}", current_plan, gr.update()

        btn_remove_step.click(
            handle_remove_step,
            inputs=[step_to_remove, steps_json_editor, state_plan],
            outputs=[steps_json_editor, plan_display, state_plan, step_to_remove],
        )

        def handle_approve(thread_id: str, current_plan: AnalysisPlan | None, steps_json: str):
            """Approve and resume the graph execution thread."""
            if not current_plan:
                return (
                    "**Status:** ⚠️ *No plan available to approve.*",
                    "",
                    "",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    {},
                    gr.update(),
                )

            # Check if steps were modified in the editor
            modified_steps = None
            if steps_json:
                try:
                    parsed = json.loads(steps_json)
                    candidate_steps = [AnalysisStep(**s) for s in parsed]
                    # If steps differ from plan, pass modified_steps
                    if candidate_steps != current_plan.steps:
                        modified_steps = candidate_steps
                except Exception as e:
                    return (
                        f"**Status:** ❌ *Edited steps JSON is invalid:* {e}",
                        "",
                        "",
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        {},
                        gr.update(),
                    )

            decision = ApprovalDecision(
                plan_id=current_plan.plan_id,
                approved=True,
                feedback="Approved via Gradio UI",
                modified_steps=modified_steps,
            )

            # Resume execution
            resumed_state = resume_research_session(
                graph=graph,
                thread_id=thread_id,
                decision=decision,
            )

            # Format execution logs
            exec_records = resumed_state.get("execution_records", [])
            log_md = format_execution_log(exec_records)

            # Format results and charts
            results = resumed_state.get("results", {})
            results_md = format_results_markdown(results)

            # Extract charts for up to 4 tools
            figs: list[go.Figure | None] = []
            for r in results.values():
                if r.chart_data and len(figs) < 4:
                    fig = convert_chart_data_to_figure(r.chart_data)
                    if fig:
                        figs.append(fig)

            plot_updates = []
            for i in range(4):
                if i < len(figs) and figs[i] is not None:
                    plot_updates.append(gr.update(value=figs[i], visible=True))
                else:
                    plot_updates.append(gr.update(visible=False))

            report = resumed_state.get("report")
            report_md = format_report_markdown(report)

            status_msg = f"**Status:** ✅ **EXECUTION COMPLETED.** Executed {len(exec_records)} analytical tools."

            return (
                status_msg,
                log_md,
                results_md,
                report_md,
                plot_updates[0],
                plot_updates[1],
                plot_updates[2],
                plot_updates[3],
                results,
                gr.update(selected="tab_results"),
            )

        btn_approve.click(
            handle_approve,
            inputs=[state_thread_id, state_plan, steps_json_editor],
            outputs=[
                status_box,
                execution_log_box,
                results_details_box,
                report_box,
                plot_1,
                plot_2,
                plot_3,
                plot_4,
                state_results,
                tabs,
            ],
        )

        def handle_reject(thread_id: str, current_plan: AnalysisPlan | None, feedback: str):
            """Reject plan and route back to planning with feedback."""
            if not current_plan:
                return (
                    "**Status:** ⚠️ *No plan available to reject.*",
                    "*No plan.*",
                    "",
                    gr.update(choices=[]),
                    None,
                    gr.update(),
                )

            decision = ApprovalDecision(
                plan_id=current_plan.plan_id,
                approved=False,
                feedback=feedback.strip() if feedback else "Plan rejected by researcher.",
            )

            resumed_state = resume_research_session(
                graph=graph,
                thread_id=thread_id,
                decision=decision,
            )

            new_plan: AnalysisPlan | None = resumed_state.get("plan")
            if new_plan:
                new_md = format_plan_markdown(new_plan)
                new_steps = [s.model_dump() for s in new_plan.steps]
                new_json = json.dumps(new_steps, indent=2)
                step_ids = [s.step_id for s in new_plan.steps]
                status_msg = (
                    f"**Status:** 🔄 *Plan rejected and regenerated with feedback.* "
                    f"Review updated plan `{new_plan.plan_id}`."
                )
                return (
                    status_msg,
                    new_md,
                    new_json,
                    gr.update(choices=step_ids, value=step_ids[0] if step_ids else None),
                    new_plan,
                    gr.update(selected="tab_plan"),
                )

            return (
                f"**Status:** *Plan rejected. Workflow ended:* {resumed_state.get('status')}",
                "*Workflow terminated.*",
                "",
                gr.update(choices=[]),
                None,
                gr.update(selected="tab_workspace"),
            )

        btn_reject.click(
            handle_reject,
            inputs=[state_thread_id, state_plan, feedback_input],
            outputs=[
                status_box,
                plan_display,
                steps_json_editor,
                step_to_remove,
                state_plan,
                tabs,
            ],
        )

        def handle_reset():
            """Reset the session with a fresh thread ID."""
            new_tid = f"session_{uuid.uuid4().hex[:8]}"
            return (
                new_tid,
                None,
                {},
                "",
                "**Status:** `New session initialized. Ready for research inquiry.`",
                gr.update(visible=False),
                "*No plan generated.*",
                "",
                gr.update(choices=[]),
                "*No tool outputs available.*",
                "*Final synthesis report not yet generated.*",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(selected="tab_workspace"),
            )

        btn_reset.click(
            handle_reset,
            inputs=None,
            outputs=[
                state_thread_id,
                state_plan,
                state_results,
                query_input,
                status_box,
                clarification_box,
                plan_display,
                steps_json_editor,
                step_to_remove,
                results_details_box,
                report_box,
                plot_1,
                plot_2,
                plot_3,
                plot_4,
                tabs,
            ],
        )

        def handle_refresh_sessions(keyword: str = ""):
            """Load session summaries from SQLite database."""
            repo = get_session_repository()
            if keyword and keyword.strip():
                summaries = repo.search_sessions(keyword.strip())
            else:
                summaries = repo.list_sessions(limit=50)

            data = []
            s_ids = []
            for s in summaries:
                ts = s["created_at"][:19].replace("T", " ")
                data.append([s["session_id"], ts, s["status"].upper(), s["query"], s["tool_count"]])
                s_ids.append(s["session_id"])

            return data, gr.update(choices=s_ids, value=s_ids[0] if s_ids else None)

        btn_refresh_sessions.click(
            handle_refresh_sessions,
            inputs=[search_input],
            outputs=[sessions_table, selected_session_dropdown],
        )

        btn_search_sessions.click(
            handle_refresh_sessions,
            inputs=[search_input],
            outputs=[sessions_table, selected_session_dropdown],
        )

        def handle_inspect_session(session_id: str):
            """Load full details of a saved session."""
            if not session_id:
                return "*No session selected.*"

            repo = get_session_repository()
            record = repo.get_session(session_id)
            if not record:
                return f"*Session '{session_id}' not found in database.*"

            sess = record["session"]
            plan = sess.plan
            report = sess.final_report

            md = [
                f"## 📂 Reopened Session: `{sess.session_id}`",
                f"- **Created At:** {sess.created_at} | **Status:** `{record['status'].upper()}`",
                f"- **Model Metadata:** Provider = `{record['model_provider']}`, Model = `{record['model_name']}`",
                f"- **Original Query:** \"{sess.request.query}\"",
                f"- **Associated Experiment:** `{sess.request.experiment_id}`",
                "",
                "### 📋 Approved Plan Details",
                format_plan_markdown(plan),
                "",
                "### 📜 Executed Operations & Audit Log",
                format_execution_log(sess.execution_records),
                "",
                "### 📊 Executed Results",
                format_results_markdown(record.get("results", {})),
                "",
                "### 📝 Final Synthesised Report",
                format_report_markdown(report),
            ]

            return "\n".join(md)

        btn_inspect_session.click(
            handle_inspect_session,
            inputs=[selected_session_dropdown],
            outputs=[session_details_markdown],
        )

    return demo


def launch_ui() -> None:
    """Launch the Gradio interface server."""
    cfg = get_settings()
    app = create_app()
    app.launch(server_name=cfg.gradio_server_name, server_port=cfg.gradio_server_port, share=False)


if __name__ == "__main__":
    launch_ui()
