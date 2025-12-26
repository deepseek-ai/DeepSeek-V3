"""
Context Window Tracking Utility for DeepSeek-V3

Provides real-time tracking and visualization of context window usage
during interactive conversations, with warnings when approaching limits.

This module addresses GitHub issue #1050: Context window limit notification system
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class ContextStatus(Enum):
    """Status levels for context window usage."""
    NORMAL = "normal"
    WARNING = "warning"      # 75% threshold
    CRITICAL = "critical"    # 90% threshold
    EXCEEDED = "exceeded"    # 100% threshold


@dataclass
class ContextMetrics:
    """Metrics for current context window state."""
    current_tokens: int
    max_tokens: int
    usage_percent: float
    status: ContextStatus


class ContextTracker:
    """
    Tracks context window usage and provides notifications when approaching limits.

    Features:
    - Real-time token count tracking
    - Visual progress bar for context usage
    - Warning thresholds at 75% and 90%
    - Summarization suggestions when nearing capacity

    Example usage:
        tracker = ContextTracker(max_seq_len=16384)
        tokens = tokenizer.apply_chat_template(messages)
        tracker.update(len(tokens))
        print(tracker.get_usage_display())

        should_warn, message = tracker.should_warn()
        if should_warn:
            print(message)
    """

    WARNING_THRESHOLD = 0.75   # 75% - first warning
    CRITICAL_THRESHOLD = 0.90  # 90% - critical warning

    def __init__(self, max_seq_len: int, bar_width: int = 30):
        """
        Initialize the context tracker.

        Args:
            max_seq_len: Maximum sequence length (context window size)
            bar_width: Width of the progress bar in characters
        """
        self.max_seq_len = max_seq_len
        self.bar_width = bar_width
        self.current_tokens = 0
        self._last_warned_status: Optional[ContextStatus] = None

    def update(self, token_count: int) -> None:
        """
        Update the current token count.

        Args:
            token_count: Current total number of tokens in the conversation
        """
        self.current_tokens = token_count

    def get_metrics(self) -> ContextMetrics:
        """
        Get current context usage metrics.

        Returns:
            ContextMetrics with current state information
        """
        usage_percent = self.current_tokens / self.max_seq_len

        if usage_percent >= 1.0:
            status = ContextStatus.EXCEEDED
        elif usage_percent >= self.CRITICAL_THRESHOLD:
            status = ContextStatus.CRITICAL
        elif usage_percent >= self.WARNING_THRESHOLD:
            status = ContextStatus.WARNING
        else:
            status = ContextStatus.NORMAL

        return ContextMetrics(
            current_tokens=self.current_tokens,
            max_tokens=self.max_seq_len,
            usage_percent=usage_percent,
            status=status
        )

    def get_usage_display(self, show_bar: bool = True) -> str:
        """
        Get a formatted string showing current context usage.

        Args:
            show_bar: Whether to include the visual progress bar

        Returns:
            Formatted string with usage information
        """
        metrics = self.get_metrics()
        percent = min(metrics.usage_percent * 100, 100)

        # Color codes for terminal output
        colors = {
            ContextStatus.NORMAL: "\033[32m",    # Green
            ContextStatus.WARNING: "\033[33m",   # Yellow
            ContextStatus.CRITICAL: "\033[31m",  # Red
            ContextStatus.EXCEEDED: "\033[91m",  # Bright Red
        }
        reset = "\033[0m"
        color = colors.get(metrics.status, "")

        parts = []

        if show_bar:
            filled = int(self.bar_width * min(metrics.usage_percent, 1.0))
            empty = self.bar_width - filled
            bar = f"[{'█' * filled}{'░' * empty}]"
            parts.append(f"{color}{bar}{reset}")

        parts.append(f"Context: {color}{metrics.current_tokens:,}/{metrics.max_tokens:,}{reset} ({percent:.1f}%)")

        return " ".join(parts)

    def should_warn(self) -> Tuple[bool, str]:
        """
        Check if a warning should be displayed to the user.

        Returns:
            Tuple of (should_warn: bool, warning_message: str)
            Only returns True once per threshold crossing to avoid spam.
        """
        metrics = self.get_metrics()

        # Don't repeat warnings for the same status level
        if metrics.status == self._last_warned_status:
            return False, ""

        # Only warn on WARNING, CRITICAL, or EXCEEDED status
        if metrics.status == ContextStatus.NORMAL:
            self._last_warned_status = None
            return False, ""

        self._last_warned_status = metrics.status

        messages = {
            ContextStatus.WARNING: (
                "\n⚠️  Context window at 75% capacity.\n"
                "💡 Tip: Consider summarizing key points to preserve important context.\n"
            ),
            ContextStatus.CRITICAL: (
                "\n🚨 Context window at 90% capacity!\n"
                "💡 Recommendation: Summarize the conversation now or start a new chat.\n"
                "   Earlier messages will be truncated to make room for new ones.\n"
            ),
            ContextStatus.EXCEEDED: (
                "\n❌ Context window limit reached!\n"
                "   The conversation history is being truncated. Some earlier context may be lost.\n"
                "💡 Use /clear to start fresh, or summarize key points to continue effectively.\n"
            ),
        }

        return True, messages.get(metrics.status, "")

    def get_summary_prompt(self) -> str:
        """
        Get a suggested prompt for summarizing the conversation.

        Returns:
            A prompt string the user can send to get a summary
        """
        return (
            "Please provide a concise summary of our conversation so far, "
            "highlighting the key points, decisions made, and any pending questions. "
            "This will help maintain context as we continue."
        )

    def reset(self) -> None:
        """Reset the tracker state (e.g., when starting a new conversation)."""
        self.current_tokens = 0
        self._last_warned_status = None

    def estimate_remaining_tokens(self) -> int:
        """
        Estimate how many tokens can still be added.

        Returns:
            Approximate number of remaining tokens available
        """
        return max(0, self.max_seq_len - self.current_tokens)

    def estimate_remaining_turns(self, avg_tokens_per_turn: int = 500) -> int:
        """
        Estimate how many conversation turns remain.

        Args:
            avg_tokens_per_turn: Average tokens per user+assistant turn

        Returns:
            Estimated number of remaining turns
        """
        remaining = self.estimate_remaining_tokens()
        return remaining // avg_tokens_per_turn


def format_token_count(count: int) -> str:
    """Format a token count with thousands separators."""
    return f"{count:,}"
