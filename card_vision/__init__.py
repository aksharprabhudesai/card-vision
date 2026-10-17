"""card-vision: Production-grade financial card detection, rectification, and attribute extraction."""

from card_vision.models import (
    BoundingBox,
    CardClass,
    CardResult,
    NetworkScheme,
    Point2D,
)
from card_vision.preprocessor import (
    STD_CARD_HEIGHT,
    STD_CARD_WIDTH,
    order_points,
    redact_pan_region,
    warp_perspective_id1,
)
from card_vision.scanner import CardScanner
from card_vision.stream import StreamCardScanner, StreamState
from card_vision.validator import (
    clean_pan_digits,
    format_pan,
    mask_pan,
    parse_expiry_date,
    resolve_bin_metadata,
    validate_luhn,
)

__version__ = "0.1.1"
__all__ = [
    "CardScanner",
    "StreamCardScanner",
    "StreamState",
    "CardResult",
    "BoundingBox",
    "Point2D",
    "NetworkScheme",
    "CardClass",
    "STD_CARD_WIDTH",
    "STD_CARD_HEIGHT",
    "validate_luhn",
    "mask_pan",
    "format_pan",
    "parse_expiry_date",
    "resolve_bin_metadata",
    "clean_pan_digits",
    "warp_perspective_id1",
    "order_points",
    "redact_pan_region",
]
