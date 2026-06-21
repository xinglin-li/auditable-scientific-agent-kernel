# src/agent_runtime/evals/dataset.py
import json
from pathlib import Path
from typing import List
from agent_runtime.evals.models import EvalTask

class EvalDatasetLoader:
    """Load evaluation datasets deterministically with strict validation."""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[EvalTask]:
        """Stream a JSONL file and validate each row as an EvalTask."""
        tasks = []
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {file_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue
                try:
                    data = json.loads(cleaned_line)
                    tasks.append(EvalTask.model_validate(data))
                except Exception as e:
                    # Identify the invalid row precisely and reject malformed data.
                    raise ValueError(f"Fatal dataset parsing error [file: {path.name} | line: {line_idx}]: {str(e)}")
                    
        return tasks
