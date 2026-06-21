# src/agent_runtime/evals/dataset.py
import json
from pathlib import Path
from typing import List
from agent_runtime.evals.models import EvalTask

class EvalDatasetLoader:
    """确定性数据集加载与强校验引擎"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[EvalTask]:
        """流式加载并逐行通过 Pydantic 校验固化为 EvalTask"""
        tasks = []
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"评估数据集文件未找到: {file_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.startswith("#"):
                    continue
                try:
                    data = json.loads(cleaned_line)
                    tasks.append(EvalTask.model_validate(data))
                except Exception as e:
                    # 强力拦截：直接指出具体哪一行发生 Schema 损坏，防止脏数据入库
                    raise ValueError(f"数据集解析致命错误 [文件: {path.name} | 第 {line_idx} 行]: {str(e)}")
                    
        return tasks