"""Risk Agent — enforces hard guardrails and checks portfolio risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from oracle.agents.base import BaseAgent
from oracle.agents.messages import Message, MessageBus, MessageType

logger = structlog.get_logger()


class RiskViolationError(Exception):
    """Raised when a hard guardrail is violated."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(f"Risk violations: {', '.join(violations)}")


@dataclass
class RiskCheckResult:
    """Result of a risk check on a trade proposal."""

    approved: bool
    violations: list[str] = field(default_factory=list)
    adjusted_size: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "violations": self.violations,
            "adjusted_size": self.adjusted_size,
            "details": self.details,
        }


# --- Hard Guardrail Constants ---
MAX_SINGLE_MARKET_PCT = 10.0  # Max 10% of portfolio in any single market
MAX_CATEGORY_EXPOSURE_PCT = 30.0  # Max 30% exposure to any single category
MAX_EXPIRING_RISK_PCT = 5.0  # Max 5% at risk on markets resolving within 24h
STOP_LOSS_THRESHOLD = 0.50  # 50% loss triggers stop-loss


class RiskAgent(BaseAgent):
    """Enforces risk guardrails on trade proposals.

    Hard guardrails (raises RiskViolationError if exceeded):
    1. Max 10% of portfolio in any single market
    2. Max 30% exposure to any single category
    3. Max 5% at risk on markets resolving within 24h
    4. Stop-loss trigger at 50% loss on any individual position
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(agent_id="risk", name="Risk Agent", bus=bus)

    async def handle_message(self, message: Message) -> None:
        """Handle RISK_CHECK messages."""
        if message.type != MessageType.RISK_CHECK:
            return

        proposal = message.payload
        trace_id = message.trace_id

        logger.info("risk.check_start", market_id=proposal.get("market_id"), trace_id=trace_id)

        result = self.check_risk(proposal)

        msg_type = MessageType.TRADE_APPROVED if result.approved else MessageType.TRADE_REJECTED

        await self.send(Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type=msg_type,
            payload={**result.to_dict(), **proposal},
            trace_id=trace_id,
        ))

    def check_risk(self, proposal: dict[str, Any]) -> RiskCheckResult:
        """Evaluate a trade proposal against all guardrails.

        Args:
            proposal: Must contain:
                - market_id: str
                - size_pct: float (% of portfolio)
                - category: str
                - portfolio: dict with {positions, cash, total_value}
                - hours_to_resolution: float | None
                - current_pnl_pct: float | None (for existing positions)
        """
        violations: list[str] = []
        requested_size = proposal.get("size_pct", 0.0)
        adjusted_size = requested_size
        details: dict[str, Any] = {}

        portfolio = proposal.get("portfolio", {})
        positions = portfolio.get("positions", {})
        total_value = portfolio.get("total_value", 10000.0)

        # --- Guardrail 1: Single market concentration ---
        market_id = proposal.get("market_id", "")
        existing_position_value = 0.0
        if market_id in positions:
            existing_position_value = positions[market_id].get("value", 0.0)

        new_exposure_pct = (
            (existing_position_value + (requested_size / 100.0 * total_value)) / total_value * 100
            if total_value > 0
            else 0
        )
        if new_exposure_pct > MAX_SINGLE_MARKET_PCT:
            violations.append(
                f"Single market exposure {new_exposure_pct:.1f}% exceeds "
                f"limit of {MAX_SINGLE_MARKET_PCT}%"
            )
            # Adjust down to fit
            max_additional = (MAX_SINGLE_MARKET_PCT / 100 * total_value) - existing_position_value
            adjusted_size = max(0, max_additional / total_value * 100) if total_value > 0 else 0
        details["single_market_exposure_pct"] = round(new_exposure_pct, 2)

        # --- Guardrail 2: Category concentration ---
        category = proposal.get("category", "other")
        category_exposure = 0.0
        for _mid, pos in positions.items():
            if pos.get("category") == category:
                category_exposure += pos.get("value", 0.0)
        new_category_pct = (
            (category_exposure + (requested_size / 100.0 * total_value)) / total_value * 100
            if total_value > 0
            else 0
        )
        if new_category_pct > MAX_CATEGORY_EXPOSURE_PCT:
            violations.append(
                f"Category '{category}' exposure {new_category_pct:.1f}% exceeds "
                f"limit of {MAX_CATEGORY_EXPOSURE_PCT}%"
            )
        details["category_exposure_pct"] = round(new_category_pct, 2)

        # --- Guardrail 3: Expiring market risk ---
        hours_to_resolution = proposal.get("hours_to_resolution")
        if hours_to_resolution is not None and hours_to_resolution < 24:
            # Sum all at-risk value in expiring markets
            expiring_risk = 0.0
            for _mid, pos in positions.items():
                if pos.get("hours_to_resolution", 999) < 24:
                    expiring_risk += pos.get("value", 0.0)
            new_expiring_pct = (
                (expiring_risk + (requested_size / 100.0 * total_value)) / total_value * 100
                if total_value > 0
                else 0
            )
            if new_expiring_pct > MAX_EXPIRING_RISK_PCT:
                violations.append(
                    f"Expiring market risk {new_expiring_pct:.1f}% exceeds "
                    f"limit of {MAX_EXPIRING_RISK_PCT}%"
                )
            details["expiring_risk_pct"] = round(new_expiring_pct, 2)

        # --- Guardrail 4: Stop-loss check (existing positions) ---
        current_pnl_pct = proposal.get("current_pnl_pct")
        if current_pnl_pct is not None and current_pnl_pct <= -STOP_LOSS_THRESHOLD * 100:
            violations.append(
                f"Stop-loss triggered: position down {abs(current_pnl_pct):.1f}% "
                f"(threshold: {STOP_LOSS_THRESHOLD * 100:.0f}%)"
            )

        approved = len(violations) == 0
        if not approved:
            adjusted_size = min(adjusted_size, requested_size)

        result = RiskCheckResult(
            approved=approved,
            violations=violations,
            adjusted_size=adjusted_size if approved else max(0, adjusted_size),
            details=details,
        )

        logger.info(
            "risk.check_complete",
            market_id=market_id,
            approved=approved,
            violations=len(violations),
        )
        return result
