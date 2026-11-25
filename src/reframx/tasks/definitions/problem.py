from dataclasses import dataclass


@dataclass(frozen = True)
class ProblemDefinition:
    name: str | None = None
    description: str | None = None
    input_semantics: str | None = None
    target_semantics: str | None = None 
    output_semantics: str | None = None
