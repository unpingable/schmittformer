from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fixed_state import GovernanceEvent, GovernanceOutput, GovernanceState
from src.stock_governance_transformer import StockGovernanceConfig, StockGovernanceTransformer, stock_step


def fmt(state: GovernanceState) -> str:
    return (
        f"auth={state.authority} lease={state.lease_remaining} "
        f"budget={state.action_budget} occurrence={state.occurrence} settlement={state.settlement}"
    )


def main() -> None:
    model = StockGovernanceTransformer(StockGovernanceConfig(score_gap=8.0, dtype_name="float32"))
    state = GovernanceState(authority=0, lease_remaining=0, action_budget=0, occurrence=0, settlement=0)
    events = [
        GovernanceEvent.GRANT_AUTHORITY,
        GovernanceEvent.RENEW_LEASE_ONE,
        GovernanceEvent.RESET_BUDGET_ONE,
        GovernanceEvent.PROPOSE_ACTION,
        GovernanceEvent.TICK,
        GovernanceEvent.RESULT_AMBIGUOUS,
        GovernanceEvent.PROPOSE_ACTION,
        GovernanceEvent.SETTLE_FAILURE,
    ]

    print("backend: stock finite-softmax + generated Linear/ReLU FFN, CPU")
    print("initial:", fmt(state))
    for event in events:
        next_state, output, state_margin, output_margin = stock_step(model, state, event)
        print(
            f"{event.name:18s} -> {GovernanceOutput(output).name:20s} "
            f"{fmt(next_state)}  margins(state={state_margin:.3f}, output={output_margin:.3f})"
        )
        state = next_state


if __name__ == "__main__":
    main()
