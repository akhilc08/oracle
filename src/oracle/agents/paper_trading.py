"""Paper Trading Engine — simulated portfolio with SQLite persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger()

DB_PATH = "oracle.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    size_pct REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    quantity REAL NOT NULL,
    value REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0.0,
    conviction_at_entry REAL DEFAULT 0.0,
    research_trace_id TEXT DEFAULT '',
    category TEXT DEFAULT 'other',
    hours_to_resolution REAL DEFAULT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT DEFAULT NULL,
    outcome TEXT DEFAULT NULL,
    realized_pnl REAL DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    size_pct REAL NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    value REAL NOT NULL,
    conviction REAL NOT NULL,
    trace_id TEXT DEFAULT '',
    executed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    pnl REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


@dataclass
class Trade:
    """Record of an executed trade."""

    id: str
    market_id: str
    direction: str  # "yes" or "no"
    size_pct: float
    price: float
    quantity: float
    value: float
    conviction: float
    trace_id: str = ""
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "market_id": self.market_id,
            "direction": self.direction,
            "size_pct": self.size_pct,
            "price": self.price,
            "quantity": self.quantity,
            "value": self.value,
            "conviction": self.conviction,
            "trace_id": self.trace_id,
            "executed_at": self.executed_at.isoformat(),
        }


class PaperTradingEngine:
    """Simulated trading engine with SQLite persistence.

    Portfolio state:
    - cash: starting at $10,000
    - positions: { market_id -> position_data }
    - total_value: cash + sum(position values)
    - pnl: total_value - initial_cash
    """

    def __init__(self, db_path: str = DB_PATH, initial_cash: float = 10000.0) -> None:
        self._db_path = db_path
        self._initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict[str, Any]] = {}

    @property
    def total_value(self) -> float:
        positions_value = sum(p.get("value", 0.0) for p in self.positions.values())
        return self.cash + positions_value

    @property
    def pnl(self) -> float:
        return self.total_value - self._initial_cash

    async def initialize(self) -> None:
        """Create tables and load existing state from SQLite."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

            # Load open positions
            async with db.execute(
                "SELECT * FROM positions WHERE closed_at IS NULL"
            ) as cursor:
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                for row in rows:
                    pos = dict(zip(cols, row))
                    self.positions[pos["market_id"]] = pos

            # Calculate cash from initial minus invested
            invested = sum(p.get("value", 0.0) for p in self.positions.values())
            # Check last portfolio_history for accurate cash
            async with db.execute(
                "SELECT cash FROM portfolio_history ORDER BY id DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    self.cash = row[0]
                else:
                    self.cash = self._initial_cash - invested

        logger.info(
            "paper_trading.initialized",
            cash=self.cash,
            positions=len(self.positions),
            total_value=self.total_value,
        )

    async def execute_trade(
        self,
        market_id: str,
        direction: str,
        size_pct: float,
        price: float,
        conviction: float,
        trace_id: str = "",
        category: str = "other",
        hours_to_resolution: float | None = None,
    ) -> Trade:
        """Execute a paper trade — buy into a market position.

        Args:
            market_id: Polymarket market ID
            direction: "yes" or "no"
            size_pct: Position size as % of total portfolio
            price: Entry price (0.0-1.0)
            conviction: Conviction score at entry
            trace_id: Research trace ID for attribution
            category: Market category
            hours_to_resolution: Hours until market resolves
        """
        trade_value = self.total_value * (size_pct / 100.0)
        quantity = trade_value / price if price > 0 else 0

        if trade_value > self.cash:
            trade_value = self.cash
            quantity = trade_value / price if price > 0 else 0
            size_pct = (trade_value / self.total_value * 100) if self.total_value > 0 else 0

        trade = Trade(
            id=uuid.uuid4().hex,
            market_id=market_id,
            direction=direction,
            size_pct=size_pct,
            price=price,
            quantity=quantity,
            value=trade_value,
            conviction=conviction,
            trace_id=trace_id,
        )

        # Update portfolio
        self.cash -= trade_value
        self.positions[market_id] = {
            "id": uuid.uuid4().hex,
            "market_id": market_id,
            "direction": direction,
            "size_pct": size_pct,
            "entry_price": price,
            "current_price": price,
            "quantity": quantity,
            "value": trade_value,
            "unrealized_pnl": 0.0,
            "conviction_at_entry": conviction,
            "research_trace_id": trace_id,
            "category": category,
            "hours_to_resolution": hours_to_resolution,
            "opened_at": trade.executed_at.isoformat(),
            "closed_at": None,
            "outcome": None,
            "realized_pnl": None,
        }

        # Persist
        await self._save_trade(trade)
        await self._save_position(self.positions[market_id])
        await self._record_portfolio_snapshot()

        logger.info(
            "paper_trading.executed",
            market_id=market_id,
            direction=direction,
            size_pct=round(size_pct, 2),
            price=price,
            value=round(trade_value, 2),
        )
        return trade

    async def close_position(self, market_id: str, outcome: str) -> dict[str, Any]:
        """Close a position when market resolves.

        Args:
            market_id: Market to close
            outcome: "yes" or "no" (the resolved outcome)

        Returns:
            Position data with realized P&L.
        """
        pos = self.positions.get(market_id)
        if pos is None:
            return {"error": f"No open position for {market_id}"}

        direction = pos["direction"]
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]

        # Settlement: if outcome matches direction, payout = quantity * 1.0, else 0
        if outcome == direction:
            settlement_value = quantity * 1.0
        else:
            settlement_value = 0.0

        cost_basis = quantity * entry_price
        realized_pnl = settlement_value - cost_basis

        # Update portfolio
        self.cash += settlement_value
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        pos["outcome"] = outcome
        pos["realized_pnl"] = realized_pnl
        pos["current_price"] = 1.0 if outcome == direction else 0.0

        # Persist updates
        await self._update_position_closed(pos)
        await self._record_portfolio_snapshot()

        # Remove from active positions
        del self.positions[market_id]

        logger.info(
            "paper_trading.closed",
            market_id=market_id,
            outcome=outcome,
            realized_pnl=round(realized_pnl, 2),
        )
        return pos

    def get_portfolio_state(self) -> dict[str, Any]:
        """Full portfolio state snapshot."""
        return {
            "cash": round(self.cash, 2),
            "positions": {
                mid: {
                    "direction": p["direction"],
                    "entry_price": p["entry_price"],
                    "current_price": p["current_price"],
                    "quantity": p["quantity"],
                    "value": p["value"],
                    "unrealized_pnl": p.get("unrealized_pnl", 0.0),
                    "conviction_at_entry": p.get("conviction_at_entry", 0.0),
                    "research_trace_id": p.get("research_trace_id", ""),
                    "category": p.get("category", "other"),
                }
                for mid, p in self.positions.items()
            },
            "total_value": round(self.total_value, 2),
            "pnl": round(self.pnl, 2),
            "position_count": len(self.positions),
        }

    # --- SQLite persistence ---

    async def _save_trade(self, trade: Trade) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO trades (id, market_id, direction, size_pct, price, "
                "quantity, value, conviction, trace_id, executed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.id, trade.market_id, trade.direction, trade.size_pct,
                    trade.price, trade.quantity, trade.value, trade.conviction,
                    trade.trace_id, trade.executed_at.isoformat(),
                ),
            )
            await db.commit()

    async def _save_position(self, pos: dict[str, Any]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO positions "
                "(id, market_id, direction, size_pct, entry_price, current_price, "
                "quantity, value, unrealized_pnl, conviction_at_entry, research_trace_id, "
                "category, hours_to_resolution, opened_at, closed_at, outcome, realized_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pos["id"], pos["market_id"], pos["direction"], pos["size_pct"],
                    pos["entry_price"], pos["current_price"], pos["quantity"], pos["value"],
                    pos.get("unrealized_pnl", 0.0), pos.get("conviction_at_entry", 0.0),
                    pos.get("research_trace_id", ""), pos.get("category", "other"),
                    pos.get("hours_to_resolution"), pos["opened_at"],
                    pos.get("closed_at"), pos.get("outcome"), pos.get("realized_pnl"),
                ),
            )
            await db.commit()

    async def _update_position_closed(self, pos: dict[str, Any]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE positions SET closed_at=?, outcome=?, realized_pnl=?, current_price=? "
                "WHERE market_id=? AND closed_at IS NULL",
                (
                    pos["closed_at"], pos["outcome"], pos["realized_pnl"],
                    pos["current_price"], pos["market_id"],
                ),
            )
            await db.commit()

    async def _record_portfolio_snapshot(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            positions_value = sum(p.get("value", 0.0) for p in self.positions.values())
            await db.execute(
                "INSERT INTO portfolio_history (total_value, cash, positions_value, pnl, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self.total_value, self.cash, positions_value,
                    self.pnl, datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
