"""Data models and schemas for card-vision."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NetworkScheme(str, Enum):
    """Payment Card Network schemes."""

    VISA = "Visa"
    MASTERCARD = "Mastercard"
    AMEX = "American Express"
    RUPAY = "RuPay"
    DISCOVER = "Discover"
    DINERS_CLUB = "Diners Club"
    JCB = "JCB"
    MAESTRO = "Maestro"
    UNKNOWN = "Unknown"


class CardClass(str, Enum):
    """Card classification based on issuing account type."""

    DEBIT = "Debit"
    CREDIT = "Credit"
    PREPAID = "Prepaid"
    UNKNOWN = "Unknown"


class Point2D(BaseModel):
    """Represents a 2D point (x, y) coordinate."""

    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")

    def to_tuple(self) -> tuple[float, float]:
        """Return point as a float tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        """Return point as an integer coordinate tuple."""
        return (int(round(self.x)), int(round(self.y)))


class BoundingBox(BaseModel):
    """Represents an axis-aligned bounding box [xmin, ymin, xmax, ymax]."""

    xmin: float = Field(..., ge=0.0, description="Top-left X coordinate")
    ymin: float = Field(..., ge=0.0, description="Top-left Y coordinate")
    xmax: float = Field(..., ge=0.0, description="Bottom-right X coordinate")
    ymax: float = Field(..., ge=0.0, description="Bottom-right Y coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")

    @field_validator("xmax")
    @classmethod
    def validate_x_bounds(cls, v: float, info: object) -> float:
        # Pydantic v2 validation
        data = getattr(info, "data", {})
        if "xmin" in data and v < data["xmin"]:
            raise ValueError("xmax must be greater than or equal to xmin")
        return v

    @field_validator("ymax")
    @classmethod
    def validate_y_bounds(cls, v: float, info: object) -> float:
        data = getattr(info, "data", {})
        if "ymin" in data and v < data["ymin"]:
            raise ValueError("ymax must be greater than or equal to ymin")
        return v

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.ymax - self.ymin

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return box as (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_int_xyxy(self) -> tuple[int, int, int, int]:
        """Return box as integer (xmin, ymin, xmax, ymax)."""
        return (int(self.xmin), int(self.ymin), int(self.xmax), int(self.ymax))


class CardResult(BaseModel):
    """Extracted attributes and verification results for a scanned financial card."""

    pan: str | None = Field(
        default=None,
        description="Full Primary Account Number (digits only). Masked if privacy requested.",
    )
    masked_pan: str | None = Field(
        default=None,
        description="Sanitized PAN formatted with masked digits (e.g. **** **** **** 1234)",
    )
    expiry_date: str | None = Field(default=None, description="Expiration date formatted as MM/YY")
    cardholder_name: str | None = Field(
        default=None, description="Sanitized cardholder name in uppercase Latin characters"
    )
    network: NetworkScheme = Field(default=NetworkScheme.UNKNOWN, description="Card network scheme")
    card_class: CardClass = Field(
        default=CardClass.UNKNOWN, description="Card class type (Credit, Debit, Prepaid)"
    )
    is_luhn_valid: bool = Field(
        default=False, description="Whether the PAN satisfies the ISO/IEC 7812 Luhn checksum"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Aggregate extraction confidence score"
    )
    corners: list[Point2D] | None = Field(
        default=None,
        description="Detected 4 corners of the card (TL, TR, BR, BL) in original image coordinates",
    )
    card_bbox: BoundingBox | None = Field(
        default=None, description="Bounding box of the card in the original image"
    )

    model_config = ConfigDict(use_enum_values=False)
