"""Validated, ordered context compaction settings for embedding hosts."""
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CompactionStep(StrEnum):
    FINISHED_RESULTS = "finished_results"
    SUMMARIZE_OLD_TURNS = "summarize_old_turns"
    READ_RESULTS = "read_results"
    FINISHED_TOOL_BATCHES = "finished_tool_batches"
    TRUNCATE_OLD_TEXT = "truncate_old_text"


class CompactionPolicy(BaseModel):
    """Percentages apply to conversation space after overhead and output reserve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trigger_percent: int = Field(default=75, strict=True, ge=1, le=100)
    target_percent: int = Field(default=50, strict=True, ge=1, le=99)
    protected_turns: int = Field(default=2, strict=True, ge=0)
    steps: tuple[CompactionStep, ...] = (
        CompactionStep.FINISHED_RESULTS,
        CompactionStep.SUMMARIZE_OLD_TURNS,
        CompactionStep.READ_RESULTS,
    )

    @model_validator(mode="after")
    def check_order(self):
        if self.target_percent >= self.trigger_percent:
            raise ValueError("target_percent must be less than trigger_percent")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("compaction steps must be unique")
        return self
