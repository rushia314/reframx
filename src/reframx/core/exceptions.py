"""Exception hierarchy for framework integration and validation."""

class ResearchFrameworkError(Exception):
        """Base exception for errors raised by Research Framework."""

class ComponentIntegrationError(ResearchFrameworkError):
    """Raised when an object cannot be integrated as a component."""

class InvalidSpecificationError(ComponentIntegrationError):
    """Raised when a component returns an invalid specification."""

class ValidatorContractError(ComponentIntegrationError):
    """Raised when a custom validator returns an unsupported result."""


class ValidationFailedError(ResearchFrameworkError):
    """Raised when experiment validation fails before execution."""

    def __init__(self, message: str, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report


class PluginLoadError(ResearchFrameworkError):
    """Raised when a plugin cannot be loaded or violates the plugin contract."""
