from dataclasses import dataclass
from typing import Literal


@dataclass(frozen = True)
class ObjectiveGoal:
    name: str
    semantics: str 
    direction: Literal[
        "minimize",
        "maximize",
        "target",
    ] = "minimize"
    description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ObjectiveGoal.name must be non-empty.")
        if not isinstance(self.semantics, str) or not self.semantics.strip():
            raise ValueError("ObjectiveGoal.semantics must be non-empty.")
        if self.direction not in ("minimize","maximize","target"):
            raise ValueError("Direction must be minimize, maximize or target.")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "semantics", self.semantics.strip())