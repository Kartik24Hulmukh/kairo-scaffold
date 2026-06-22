"""G2 — Runtime error handling and graceful degradation for the Kairo sidecar.

Defines a typed exception hierarchy so callers can distinguish recoverable
conditions (bad grounding, missing index, unreachable sidecar) from
unexpected failures, and format them into a stable dict for the UI layer.
"""


class KairoError(Exception):
    """Base class for all expected Kairo runtime errors.

    Any KairoError subclass is considered recoverable — the system can
    surface a user-facing message and continue operating.
    """


class GroundingError(KairoError):
    """Raised when a grounding verification step cannot anchor an extraction.

    The caller should mark the field as 'blocked' rather than crashing.
    """


class IndexError(KairoError):
    """Raised when a document indexing operation fails.

    Distinct from Python's built-in IndexError; always import by full path
    or alias when both are in scope.
    """


class SidecarUnavailable(KairoError):
    """Raised when the Kairo sidecar process cannot be reached.

    Callers should fall back to cached state or surface an offline notice.
    """


def format_user_error(exc: Exception) -> dict:
    """Format an exception into a stable dict for display in the UI layer.

    Returns a dict with keys:
      - error_type (str): class name of the exception
      - message (str): str(exc)
      - recoverable (bool): True for KairoError subclasses, False otherwise

    This function never raises; unknown exception types yield recoverable=False.
    """
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "recoverable": isinstance(exc, KairoError),
    }
