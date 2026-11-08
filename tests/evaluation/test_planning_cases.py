"""Evaluation test suite covering the 10 representative research planning requests from Section 10."""

import pytest
from neuroflow.agent.planner import AnalysisPlanner


class TestDemoEvaluationSet:
    """Evaluates the 10 representative research requests defined in Section 10 of README."""

    @pytest.fixture
    def planner(self) -> AnalysisPlanner:
        return AnalysisPlanner()

    def test_case_01_compare_groups(self, planner: AnalysisPlanner) -> None:
        """Case 1: Compare memory accuracy between two groups."""
        query = "Compare memory accuracy between the control and experimental groups."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "compare_experimental_groups" in tool_names
        assert len(result.plan.steps) >= 2

    def test_case_02_learning_progression(self, planner: AnalysisPlanner) -> None:
        """Case 2: Examine learning progression over trials."""
        query = "Examine learning progression over repeated trials for participants."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "analyze_learning_curve" in tool_names

    def test_case_03_response_time_error_rate_correlation(self, planner: AnalysisPlanner) -> None:
        """Case 3: Investigate response time and error rate association."""
        query = "Investigate whether response time is associated with error rate."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "analyze_performance_correlations" in tool_names

    def test_case_04_task_difficulty(self, planner: AnalysisPlanner) -> None:
        """Case 4: Identify tasks with the lowest accuracy."""
        query = "Identify which memory tasks have the lowest accuracy and highest difficulty."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "analyze_task_difficulty" in tool_names

    def test_case_05_anomalous_sessions(self, planner: AnalysisPlanner) -> None:
        """Case 5: Find unusual or incomplete sessions."""
        query = "Find unusual or anomalous sessions with high latency or incomplete blocks."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "detect_anomalous_sessions" in tool_names

    def test_case_06_behavioural_clustering(self, planner: AnalysisPlanner) -> None:
        """Case 6: Cluster participants by behavioural features."""
        query = "Cluster participants by behavioural strategy and response features."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "cluster_participant_behaviour" in tool_names

    def test_case_07_supervised_prediction(self, planner: AnalysisPlanner) -> None:
        """Case 7: Predict a predefined task outcome."""
        query = "Predict high task accuracy classification based on response time and attempt count."
        result = planner.plan(query)

        assert result.status == "plan_generated"
        assert result.plan is not None
        tool_names = [s.tool_name for s in result.plan.steps]
        assert "validate_experiment_data" in tool_names
        assert "predict_task_performance" in tool_names

    def test_case_08_ambiguous_question_clarification(self, planner: AnalysisPlanner) -> None:
        """Case 8: Ask an ambiguous question that requires clarification."""
        query = "do some analysis on the participants"
        result = planner.plan(query)

        assert result.status == "clarification_needed"
        assert result.clarification is not None
        assert result.clarification.needs_clarification is True
        assert len(result.clarification.options) > 0

    def test_case_09_unknown_variable_rejection(self, planner: AnalysisPlanner) -> None:
        """Case 9: Ask about a variable that is not in the data dictionary."""
        query = "Compare participant blood_pressure and galvanic response between groups."
        result = planner.plan(query)

        assert result.status == "clarification_needed"
        assert result.clarification is not None
        assert len(result.clarification.missing_variables) > 0
        assert len(result.clarification.suggested_alternatives) > 0

    def test_case_10_unsupported_analysis_rejection(self, planner: AnalysisPlanner) -> None:
        """Case 10: Request an unsupported analysis and verify system refuses or offers alternatives."""
        query = "Train a custom deep neural network in PyTorch to diagnose cognitive impairment."
        result = planner.plan(query)

        assert result.status == "clarification_needed"
        assert result.clarification is not None
        assert len(result.clarification.unsupported_features) > 0
        assert len(result.clarification.suggested_alternatives) > 0
