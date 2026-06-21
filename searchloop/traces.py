from __future__ import annotations

import json
from dataclasses import dataclass, field

from searchloop.env import State


def state_features(state: State, max_steps: int) -> list[float]:
    return [
        float(len(state.evidence)),
        float(state.steps),
        float(max_steps - state.steps),
        float(state.resolved),
        float(len(state.observations)),
        float(sum(1 for observation in state.observations if not observation.result.ok)),
        float(len({observation.action.tool for observation in state.observations})),
    ]


@dataclass
class TraceCollector:
    records: list[tuple[list[float], float]] = field(default_factory=list)

    def record(self, features: list[float], value: float) -> None:
        self.records.append((features, value))

    def __len__(self) -> int:
        return len(self.records)


def write_traces(path: str, records) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for features, value in records:
            file.write(json.dumps({"features": features, "value": value}) + "\n")


def read_traces(path: str) -> list[tuple[list[float], float]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(([float(value) for value in item["features"]], float(item["value"])))
    return records
