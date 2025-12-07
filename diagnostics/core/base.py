"""
Base classes for the PRISM diagnostics framework.

Provides the foundational abstractions for creating diagnostic checks
that can be registered and run through the unified diagnostic runner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import traceback


class DiagnosticStatus(Enum):
    """Status codes for diagnostic results."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"
    SKIP = "skip"
    TIMEOUT = "timeout"

    def __str__(self):
        return self.value

    @property
    def symbol(self) -> str:
        """Return a symbol for display."""
        symbols = {
            DiagnosticStatus.PASS: "[PASS]",
            DiagnosticStatus.FAIL: "[FAIL]",
            DiagnosticStatus.WARN: "[WARN]",
            DiagnosticStatus.ERROR: "[ERROR]",
            DiagnosticStatus.SKIP: "[SKIP]",
            DiagnosticStatus.TIMEOUT: "[TIMEOUT]",
        }
        return symbols.get(self, "[?]")

    @property
    def is_success(self) -> bool:
        """Check if status indicates success."""
        return self in (DiagnosticStatus.PASS, DiagnosticStatus.WARN, DiagnosticStatus.SKIP)


class DiagnosticCategory(Enum):
    """Categories of diagnostics."""
    HEALTH = "health"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    DATA = "data"
    ENGINE = "engine"
    SYSTEM = "system"

    def __str__(self):
        return self.value


@dataclass
class DiagnosticResult:
    """
    Container for diagnostic check results.

    Attributes:
        name: Name of the diagnostic
        status: Pass/fail/error status
        message: Human-readable result message
        category: Diagnostic category
        duration_ms: Time taken to run in milliseconds
        details: Additional structured data
        timestamp: When the diagnostic was run
        suggestions: List of suggested fixes if failed
    """
    name: str
    status: DiagnosticStatus
    message: str
    category: DiagnosticCategory
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    suggestions: List[str] = field(default_factory=list)
    error_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'category': self.category.value,
            'duration_ms': round(self.duration_ms, 2),
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'suggestions': self.suggestions,
            'error_trace': self.error_trace,
        }

    def __str__(self) -> str:
        return f"{self.status.symbol} {self.name}: {self.message}"


class BaseDiagnostic(ABC):
    """
    Abstract base class for all diagnostics.

    Subclass this to create custom diagnostics. Implement the `check()` method
    to perform the actual diagnostic logic.

    Example:
        class DatabaseConnectionDiagnostic(BaseDiagnostic):
            name = "database_connection"
            description = "Check database connectivity"
            category = DiagnosticCategory.HEALTH

            def check(self) -> DiagnosticResult:
                try:
                    # ... check connection ...
                    return self.pass_result("Database connected successfully")
                except Exception as e:
                    return self.fail_result(f"Connection failed: {e}")
    """

    # Override these in subclasses
    name: str = "base_diagnostic"
    description: str = "Base diagnostic check"
    category: DiagnosticCategory = DiagnosticCategory.HEALTH
    timeout_seconds: float = 60.0
    is_quick: bool = True  # If False, skipped in quick mode
    dependencies: List[str] = []  # Names of diagnostics this depends on

    def __init__(self):
        self._start_time: Optional[float] = None

    @abstractmethod
    def check(self) -> DiagnosticResult:
        """
        Perform the diagnostic check.

        Returns:
            DiagnosticResult with status and details
        """
        pass

    def run(self) -> DiagnosticResult:
        """
        Execute the diagnostic with timing and error handling.

        This wraps the check() method with:
        - Timing measurement
        - Exception handling
        - Timeout protection (if implemented by runner)
        """
        self._start_time = time.time()
        try:
            result = self.check()
            result.duration_ms = self._elapsed_ms()
            return result
        except Exception as e:
            return DiagnosticResult(
                name=self.name,
                status=DiagnosticStatus.ERROR,
                message=f"Diagnostic failed with exception: {str(e)}",
                category=self.category,
                duration_ms=self._elapsed_ms(),
                error_trace=traceback.format_exc(),
                suggestions=["Check the error trace for details", "Ensure all dependencies are installed"],
            )

    def _elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000

    # Helper methods for creating results

    def pass_result(
        self,
        message: str,
        details: Dict[str, Any] = None,
    ) -> DiagnosticResult:
        """Create a passing result."""
        return DiagnosticResult(
            name=self.name,
            status=DiagnosticStatus.PASS,
            message=message,
            category=self.category,
            details=details or {},
        )

    def fail_result(
        self,
        message: str,
        details: Dict[str, Any] = None,
        suggestions: List[str] = None,
    ) -> DiagnosticResult:
        """Create a failing result."""
        return DiagnosticResult(
            name=self.name,
            status=DiagnosticStatus.FAIL,
            message=message,
            category=self.category,
            details=details or {},
            suggestions=suggestions or [],
        )

    def warn_result(
        self,
        message: str,
        details: Dict[str, Any] = None,
        suggestions: List[str] = None,
    ) -> DiagnosticResult:
        """Create a warning result."""
        return DiagnosticResult(
            name=self.name,
            status=DiagnosticStatus.WARN,
            message=message,
            category=self.category,
            details=details or {},
            suggestions=suggestions or [],
        )

    def skip_result(self, reason: str) -> DiagnosticResult:
        """Create a skipped result."""
        return DiagnosticResult(
            name=self.name,
            status=DiagnosticStatus.SKIP,
            message=f"Skipped: {reason}",
            category=self.category,
        )

    def error_result(
        self,
        message: str,
        error_trace: str = None,
    ) -> DiagnosticResult:
        """Create an error result."""
        return DiagnosticResult(
            name=self.name,
            status=DiagnosticStatus.ERROR,
            message=message,
            category=self.category,
            error_trace=error_trace,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"
