"""Pydantic schemas — single source of truth for the agent I/O contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal[
    "load_prices",
    "compute_returns",
    "compute_volatility",
    "moving_average",
    "simple_forecast",
]


class PlanStep(BaseModel):
    id: str = Field(..., description="Unique step identifier, e.g. 's1', 's2'")
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of steps whose output this step needs",
    )
    rationale: str = Field(..., description="Why this step is needed — used by the alignment judge")


class Plan(BaseModel):
    steps: list[PlanStep]


class NumericClaim(BaseModel):
    statement: str = Field(..., description="Human-readable description of the claim")
    value: float
    source_step: str = Field(..., description="Step ID whose observation produced this value")


class FinalAnswer(BaseModel):
    summary: str = Field(..., description="Narrative answer to the user query")
    claims: list[NumericClaim] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
