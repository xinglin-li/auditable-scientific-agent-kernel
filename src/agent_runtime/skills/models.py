# src/agent_runtime/skills/models.py
from pydantic import BaseModel, Field
from typing import List, Dict

class SkillMetadata(BaseModel):
    name: str
    description: str
    root_path: str
    allowed_tools: List[str]

class ActivatedSkill(BaseModel):
    name: str
    metadata: SkillMetadata
    full_instructions: str