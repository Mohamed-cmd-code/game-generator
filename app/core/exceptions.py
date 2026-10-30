"""Domain exceptions. The API layer maps them to HTTP status codes."""


class GameGeneratorError(Exception):
    """Base class for all application errors."""


class LLMUnavailableError(GameGeneratorError):
    """The LLM provider could not be reached or returned a transport error (-> HTTP 503)."""


class LLMConfigurationError(GameGeneratorError):
    """The LLM is not configured (e.g. missing GROQ_API_KEY) (-> HTTP 503)."""


class NotFoundError(GameGeneratorError):
    """A referenced resource (game, player) does not exist (-> HTTP 404)."""
