from dataclasses import dataclass


@dataclass(frozen = True)
class EvaluationDefinition:
    primary_metrics: tuple[str, ...] = ()
    secondary_metrics: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        primary = tuple(self.primary_metrics)
        secondary = tuple(self.secondary_metrics)

        if not all(isinstance(x,str) and x.strip() for x in primary):
            raise ValueError("primary_metrics must contain non empty strings.")
        if not all(isinstance(x,str) and x.strip() for x in secondary):
            raise ValueError("secondary_metrics must contain non empty strings.")

        object.__setattr__(self, "primary_metrics", tuple(x.strip() for x in primary))
        object.__setattr__(self, "secondary_metrics", tuple(x.strip() for x in secondary))