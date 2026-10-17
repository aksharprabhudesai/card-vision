"""Multi-frame temporal stabilization and voting engine for video/webcam feeds."""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from card_vision.models import CardResult, Point2D
from card_vision.scanner import CardScanner


@dataclass
class StreamState:
    """State tracking across consecutive video frames."""

    history_size: int = 5
    min_consistent_frames: int = 3
    pan_history: list[str] = field(default_factory=list)
    expiry_history: list[str] = field(default_factory=list)
    name_history: list[str] = field(default_factory=list)
    recent_results: list[CardResult] = field(default_factory=list)
    smoothed_corners: list[Point2D] | None = None

    def reset(self) -> None:
        """Reset temporal buffers."""
        self.pan_history.clear()
        self.expiry_history.clear()
        self.name_history.clear()
        self.recent_results.clear()
        self.smoothed_corners = None


class StreamCardScanner:
    """Stream-optimized scanner with temporal consensus voting and corner smoothing.

    Prevents OCR glitches caused by temporary glare, motion blur, or camera autofocus
    by aggregating predictions across multiple successive frames.
    """

    def __init__(
        self,
        scanner: CardScanner | None = None,
        history_size: int = 5,
        min_consistent_frames: int = 3,
        smoothing_factor: float = 0.35,
    ) -> None:
        """Initialize the stream scanner.

        Args:
            scanner: Underlying CardScanner instance. If None, instantiates default.
            history_size: Number of recent frames to retain for majority voting.
            min_consistent_frames: Minimum matching consensus votes required before confirming.
            smoothing_factor: Exponential smoothing weight for corner jitter reduction (0.0 to 1.0).
        """
        self.scanner = scanner or CardScanner()
        self.history_size = history_size
        self.min_consistent_frames = min_consistent_frames
        self.smoothing_factor = smoothing_factor
        self.state = StreamState(
            history_size=history_size,
            min_consistent_frames=min_consistent_frames,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        mask_pan_flag: bool = True,
    ) -> tuple[CardResult | None, bool]:
        """Process a single frame from a video stream.

        Args:
            frame: Video frame (BGR numpy array).
            mask_pan_flag: Whether to mask PAN digits in result.

        Returns:
            Tuple of (Consensus CardResult or latest result, is_stable_boolean).
            is_stable is True when consensus criteria across consecutive frames are satisfied.
        """
        raw_result = self.scanner.scan(frame, mask_pan_flag=mask_pan_flag)
        if raw_result is None or not raw_result.is_luhn_valid:
            # Drop older history on prolonged loss of card
            if len(self.state.recent_results) > 0:
                self.state.recent_results.pop(0)
            return raw_result, False

        # Corner temporal exponential moving average (EMA)
        if raw_result.corners and len(raw_result.corners) == 4:
            if self.state.smoothed_corners is None:
                self.state.smoothed_corners = [Point2D(x=p.x, y=p.y) for p in raw_result.corners]
            else:
                alpha = self.smoothing_factor
                for i in range(4):
                    curr = raw_result.corners[i]
                    prev = self.state.smoothed_corners[i]
                    self.state.smoothed_corners[i] = Point2D(
                        x=alpha * curr.x + (1.0 - alpha) * prev.x,
                        y=alpha * curr.y + (1.0 - alpha) * prev.y,
                    )
            raw_result.corners = self.state.smoothed_corners

        # Append attributes to temporal histories
        if raw_result.pan:
            self.state.pan_history.append(raw_result.pan)
        if raw_result.expiry_date:
            self.state.expiry_history.append(raw_result.expiry_date)
        if raw_result.cardholder_name:
            self.state.name_history.append(raw_result.cardholder_name)
        self.state.recent_results.append(raw_result)

        # Trim sliding windows
        if len(self.state.pan_history) > self.history_size:
            self.state.pan_history.pop(0)
        if len(self.state.expiry_history) > self.history_size:
            self.state.expiry_history.pop(0)
        if len(self.state.name_history) > self.history_size:
            self.state.name_history.pop(0)
        if len(self.state.recent_results) > self.history_size:
            self.state.recent_results.pop(0)

        # Majority vote resolution
        is_stable = False
        pan_counts = Counter(self.state.pan_history)
        if pan_counts:
            most_common_pan, pan_freq = pan_counts.most_common(1)[0]
            if pan_freq >= self.min_consistent_frames:
                is_stable = True
                # Construct consensus CardResult
                consensus = raw_result.model_copy()
                consensus.pan = most_common_pan
                if raw_result.masked_pan and most_common_pan:
                    consensus.masked_pan = raw_result.masked_pan

                if self.state.expiry_history:
                    exp_counts = Counter(self.state.expiry_history)
                    best_exp, exp_freq = exp_counts.most_common(1)[0]
                    if exp_freq >= 2:
                        consensus.expiry_date = best_exp

                if self.state.name_history:
                    name_counts = Counter(self.state.name_history)
                    best_name, name_freq = name_counts.most_common(1)[0]
                    if name_freq >= 2:
                        consensus.cardholder_name = best_name

                consensus.confidence = min(1.0, raw_result.confidence + 0.05)
                return consensus, True

        return raw_result, is_stable

    def scan_video_stream(
        self,
        frame_generator: Iterator[np.ndarray],
        on_frame_processed: Callable[[np.ndarray, CardResult | None, bool], None] | None = None,
        stop_on_stable: bool = True,
        mask_pan_flag: bool = True,
    ) -> CardResult | None:
        """Consume a frame generator until a stable consensus card is verified.

        Args:
            frame_generator: Iterator yielding BGR image arrays (e.g. from cv2.VideoCapture).
            on_frame_processed: Optional callback(frame, current_result, is_stable) for GUI/HUD rendering.
            stop_on_stable: If True, terminates immediately upon achieving consensus.
            mask_pan_flag: Whether to mask PAN in results.

        Returns:
            Stable consensus CardResult, or None if stream terminates without consensus.
        """
        self.state.reset()
        for frame in frame_generator:
            result, is_stable = self.process_frame(frame, mask_pan_flag=mask_pan_flag)
            if on_frame_processed:
                on_frame_processed(frame, result, is_stable)
            if stop_on_stable and is_stable:
                return result
        return None
