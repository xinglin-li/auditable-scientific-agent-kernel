# src/agent_runtime/retrieval/tokenizer.py
import re
from typing import List


class SimpleTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []

        lowered = text.lower()
        cleaned = re.sub(r"[^\w\s\(\),-]", "", lowered)

        tokens = []
        for token in cleaned.split():
            if not token:
                continue

            tokens.append(token)

            if "-" in token:
                parts = [part for part in token.split("-") if part]
                tokens.extend(parts)

        return tokens
