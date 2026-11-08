"""Session repository for persisting and reconstructing research analysis runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuroflow.persistence.database import get_connection, init_db
from neuroflow.schemas.analysis import AnalysisResult
from neuroflow.schemas.planning import AnalysisPlan, ResearchRequest
from neuroflow.schemas.session import (
    ApprovalDecision,
    ExecutionRecord,
    FinalReport,
    ResearchSession,
)


class SessionRepository:
    """Repository for storing and retrieving research sessions in SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = db_path
        init_db(self.db_path)

    def save_session(
        self,
        session: ResearchSession,
        model_provider: str | None = None,
        model_name: str | None = None,
        results: dict[str, AnalysisResult] | None = None,
        status: str = "completed",
    ) -> None:
        """Persist or update a research session record."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        results_dict = (
            {k: v.model_dump() for k, v in results.items()} if results else {}
        )

        cursor.execute(
            """
            INSERT INTO sessions (
                session_id, created_at, updated_at, status, query, experiment_id,
                model_provider, model_name, plan_json, approval_json,
                execution_records_json, results_json, final_report_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                status = excluded.status,
                plan_json = excluded.plan_json,
                approval_json = excluded.approval_json,
                execution_records_json = excluded.execution_records_json,
                results_json = excluded.results_json,
                final_report_json = excluded.final_report_json,
                metadata_json = excluded.metadata_json
            """,
            (
                session.session_id,
                session.created_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                status,
                session.request.query,
                session.request.experiment_id,
                model_provider or "default",
                model_name or "mock-model",
                session.plan.model_dump_json() if session.plan else None,
                session.approval.model_dump_json() if session.approval else None,
                json.dumps([r.model_dump() for r in session.execution_records], default=str),
                json.dumps(results_dict, default=str),
                session.final_report.model_dump_json() if session.final_report else None,
                json.dumps({"version": "0.1.0"}, default=str),
            ),
        )

        conn.commit()
        conn.close()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve full details of a persisted research session."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Reconstruct into structured dict with parsed objects
        plan_data = json.loads(row["plan_json"]) if row["plan_json"] else None
        plan = AnalysisPlan(**plan_data) if plan_data else None

        approval_data = json.loads(row["approval_json"]) if row["approval_json"] else None
        approval = ApprovalDecision(**approval_data) if approval_data else None

        exec_data = json.loads(row["execution_records_json"]) if row["execution_records_json"] else []
        execution_records = [ExecutionRecord(**r) for r in exec_data]

        report_data = json.loads(row["final_report_json"]) if row["final_report_json"] else None
        report = FinalReport(**report_data) if report_data else None

        results_data = json.loads(row["results_json"]) if row["results_json"] else {}
        results = {k: AnalysisResult(**v) for k, v in results_data.items()}

        req = ResearchRequest(
            request_id=f"req_{session_id}",
            query=row["query"],
            experiment_id=row["experiment_id"],
        )

        session = ResearchSession(
            session_id=row["session_id"],
            request=req,
            plan=plan,
            approval=approval,
            execution_records=execution_records,
            final_report=report,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

        return {
            "session": session,
            "results": results,
            "status": row["status"],
            "model_provider": row["model_provider"],
            "model_name": row["model_name"],
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return a summary list of all stored research sessions."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT session_id, created_at, status, query, experiment_id, model_provider, model_name, execution_records_json
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        summaries = []
        for r in rows:
            exec_records = json.loads(r["execution_records_json"]) if r["execution_records_json"] else []
            summaries.append(
                {
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "status": r["status"],
                    "query": r["query"],
                    "experiment_id": r["experiment_id"],
                    "model_provider": r["model_provider"],
                    "model_name": r["model_name"],
                    "tool_count": len(exec_records),
                }
            )

        return summaries

    def search_sessions(self, keyword: str) -> list[dict[str, Any]]:
        """Search sessions by query keyword or session ID."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        pattern = f"%{keyword.strip()}%"
        cursor.execute(
            """
            SELECT session_id, created_at, status, query, experiment_id, model_provider, model_name, execution_records_json
            FROM sessions
            WHERE query LIKE ? OR session_id LIKE ?
            ORDER BY created_at DESC
            """,
            (pattern, pattern),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            exec_records = json.loads(r["execution_records_json"]) if r["execution_records_json"] else []
            results.append(
                {
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "status": r["status"],
                    "query": r["query"],
                    "tool_count": len(exec_records),
                }
            )
        return results

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from persistence."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


# Default repository instance
_default_repository = SessionRepository()


def get_session_repository() -> SessionRepository:
    """Return the global default SessionRepository."""
    return _default_repository
