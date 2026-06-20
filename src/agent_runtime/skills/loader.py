# src/agent_runtime/skills/loader.py
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent_runtime.skills.models import ActivatedSkill, SkillMetadata


class SkillLoader:
    def __init__(self, skills_root: str = "skills"):
        self.root_path = Path(skills_root)

    def _parse_frontmatter(self, file_content: str) -> Tuple[Dict[str, Any], str]:
        """Parse simple YAML-style frontmatter wrapped in leading --- markers."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.search(pattern, file_content, re.DOTALL)
        if not match:
            return {}, file_content

        yaml_block = match.group(1)
        body_content = match.group(2)

        meta = {}
        for line in yaml_block.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                cleaned_v = v.strip().strip('"').strip("'")
                if cleaned_v.startswith("[") and cleaned_v.endswith("]"):
                    cleaned_v = [x.strip().strip('"').strip("'") for x in cleaned_v[1:-1].split(",") if x]
                meta[k.strip()] = cleaned_v
        return meta, body_content

    def discover_catalog(self) -> List[SkillMetadata]:
        """Stage 1: discover only lightweight skill metadata, not full SOP bodies."""
        catalog = []
        if not self.root_path.exists():
            return []

        for folder in self.root_path.iterdir():
            if folder.is_dir():
                skill_file = folder / "SKILL.md"
                if skill_file.exists():
                    with open(skill_file, "r", encoding="utf-8") as f:
                        meta, _ = self._parse_frontmatter(f.read())
                        if meta and "name" in meta and "description" in meta:
                            catalog.append(
                                SkillMetadata(
                                    name=meta["name"],
                                    description=meta["description"],
                                    root_path=str(folder),
                                    allowed_tools=meta.get("allowed_tools", []),
                                )
                            )
        return catalog

    def load_full_skill(self, meta: SkillMetadata) -> ActivatedSkill:
        """Stage 2: load the full skill body only after activation."""
        skill_file = Path(meta.root_path) / "SKILL.md"
        with open(skill_file, "r", encoding="utf-8") as f:
            _, body = self._parse_frontmatter(f.read())
            return ActivatedSkill(
                name=meta.name,
                metadata=meta,
                full_instructions=body.strip(),
            )
