# src/agent_runtime/context/budget.py


class TokenBudgetTracker:
    @staticmethod
    def estimate_tokens(content: str) -> int:
        """Deterministic token estimate: roughly four characters per token, minimum one token."""
        if not content:
            return 1
        return max(1, len(content) // 4)
