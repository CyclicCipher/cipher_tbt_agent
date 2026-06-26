"""Perception — the ONLY task-format-aware code: turn an environment's observation (an ARC frame, …) into the
column's symbolic inputs (the body, objects, transitions, the score-derived goal). The agent and the column
never see a raw frame; they see what perception emits. Keeping the format knowledge here is what lets the agent
stay a thin, reusable shell (see feedback_thin_shell_agent / REORG_PLAN.md)."""
