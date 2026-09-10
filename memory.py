"""Week 4: agent memory.

ShortTermMemory is a per-run scratchpad of what happened during a single
ReAct loop; it lives only in-process and is discarded when the run ends.

LongTermMemory persists across runs in a JSON file: analyst preferences
(set once via `--remember`, applied to every future run) and per-company
notes. Notes accumulate automatically at the end of each run (an excerpt of
the draft's Executive Summary) or explicitly whenever the agent calls the
save_memory_note tool. This gives later runs prior context beyond what's in
the static sample documents under data/.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

MEMORY_PATH = Path(__file__).parent / "memory" / "agent_memory.json"


@dataclass
class ShortTermMemory:
    entries: list[str] = field(default_factory=list)

    def add(self, entry: str) -> None:
        self.entries.append(entry)

    def transcript(self) -> str:
        return "\n".join(f"- {entry}" for entry in self.entries) if self.entries else "(empty)"


@dataclass
class LongTermMemory:
    preferences: list[str] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    path: Path = field(default=MEMORY_PATH, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path = MEMORY_PATH) -> "LongTermMemory":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(preferences=data.get("preferences", []), notes=data.get("notes", []), path=path)

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(
            json.dumps({"preferences": self.preferences, "notes": self.notes}, indent=2),
            encoding="utf-8",
        )

    def add_preference(self, preference: str) -> None:
        if preference not in self.preferences:
            self.preferences.append(preference)
            self.save()

    def add_note(self, company: str, note: str) -> None:
        self.notes.append({"company": company, "date": date.today().isoformat(), "note": note})
        self.save()

    def notes_for(self, company: str, limit: int = 5) -> list[dict]:
        matches = [n for n in self.notes if n["company"].lower() == company.lower()]
        return matches[-limit:]

    def context_summary(self, company: str) -> str:
        lines = []
        if self.preferences:
            lines.append("Analyst preferences (always apply these):")
            lines.extend(f"- {p}" for p in self.preferences)
        prior = self.notes_for(company)
        if prior:
            lines.append(f"Prior context remembered about {company}:")
            lines.extend(f"- [{n['date']}] {n['note']}" for n in prior)
        return "\n".join(lines) if lines else "(no long-term memory yet for this analyst/company)"
