"""
Dimensional Analysis Module
===========================

Verify dimensional consistency of equations and derive scaling laws.
Critical for physics rigor and avoiding nonsensical combinations.

Key principles:
- Quantities must have consistent dimensions
- Equations must be dimensionally homogeneous
- Dimensionless combinations carry physical meaning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Dimension(Enum):
    """Base dimensions for financial/economic quantities."""
    PRICE = 'price'           # Currency units
    TIME = 'time'             # Time units
    RATE = 'rate'             # Per-time (1/T)
    RATIO = 'ratio'           # Dimensionless ratio
    VOLATILITY = 'volatility' # Price/sqrt(Time)
    COUNT = 'count'           # Dimensionless count
    PERCENT = 'percent'       # Dimensionless percentage


@dataclass
class DimensionalQuantity:
    """A quantity with its dimensional specification."""
    name: str
    value: float
    dimension: Dimension
    unit: str
    notes: str = ""


@dataclass
class DimensionalConsistencyResult:
    """Result of dimensional consistency check."""
    expression: str
    is_consistent: bool
    left_dimension: str
    right_dimension: str
    issue: Optional[str] = None


@dataclass
class DimensionalReport:
    """Complete dimensional analysis report."""
    quantities: List[DimensionalQuantity]
    consistency_checks: List[DimensionalConsistencyResult]
    dimensionless_groups: List[Dict[str, Any]]
    scaling_laws: List[str]
    issues: List[str]


class DimensionalAnalyzer:
    """
    Perform dimensional analysis on quantities and equations.
    """

    # Standard dimensions for common financial quantities
    STANDARD_DIMENSIONS = {
        # Prices and values
        'price': Dimension.PRICE,
        'value': Dimension.PRICE,
        'wealth': Dimension.PRICE,
        'capital': Dimension.PRICE,

        # Returns and rates
        'return': Dimension.RATIO,
        'log_return': Dimension.RATIO,
        'interest_rate': Dimension.RATE,
        'yield': Dimension.RATE,
        'growth_rate': Dimension.RATE,

        # Volatility
        'volatility': Dimension.VOLATILITY,
        'std': Dimension.VOLATILITY,
        'variance': Dimension.RATIO,  # vol^2, but per time

        # Ratios
        'ratio': Dimension.RATIO,
        'correlation': Dimension.RATIO,
        'sharpe': Dimension.RATIO,
        'hurst': Dimension.RATIO,
        'coherence': Dimension.RATIO,

        # Time
        'time': Dimension.TIME,
        'duration': Dimension.TIME,
        'maturity': Dimension.TIME,

        # Other
        'count': Dimension.COUNT,
        'percent': Dimension.PERCENT,
    }

    def __init__(self):
        self.quantities: Dict[str, DimensionalQuantity] = {}

    def assign_dimension(self,
                          name: str,
                          value: float,
                          dimension: str,
                          unit: str = "") -> DimensionalQuantity:
        """
        Assign dimension to a quantity.

        Args:
            name: Quantity name
            value: Numeric value
            dimension: Dimension name or Dimension enum
            unit: Unit string (e.g., 'USD', 'days')

        Returns:
            DimensionalQuantity
        """
        if isinstance(dimension, str):
            if dimension.lower() in self.STANDARD_DIMENSIONS:
                dim = self.STANDARD_DIMENSIONS[dimension.lower()]
            else:
                dim = Dimension.RATIO  # Default to dimensionless
        else:
            dim = dimension

        qty = DimensionalQuantity(
            name=name,
            value=value,
            dimension=dim,
            unit=unit
        )

        self.quantities[name] = qty
        return qty

    def check_consistency(self,
                           left: str,
                           right: str,
                           operation: str = '=') -> DimensionalConsistencyResult:
        """
        Check if two quantities can be combined.

        Args:
            left: Left quantity name
            right: Right quantity name
            operation: '=', '+', '-', '*', '/'

        Returns:
            DimensionalConsistencyResult
        """
        left_qty = self.quantities.get(left)
        right_qty = self.quantities.get(right)

        if left_qty is None:
            return DimensionalConsistencyResult(
                expression=f"{left} {operation} {right}",
                is_consistent=False,
                left_dimension="unknown",
                right_dimension=right_qty.dimension.value if right_qty else "unknown",
                issue=f"Unknown quantity: {left}"
            )

        if right_qty is None:
            return DimensionalConsistencyResult(
                expression=f"{left} {operation} {right}",
                is_consistent=False,
                left_dimension=left_qty.dimension.value,
                right_dimension="unknown",
                issue=f"Unknown quantity: {right}"
            )

        left_dim = left_qty.dimension
        right_dim = right_qty.dimension

        if operation in ['=', '+', '-']:
            # Must have same dimension
            is_consistent = left_dim == right_dim
            issue = None if is_consistent else "Dimensions must match for equality/addition"

        elif operation == '*':
            # Multiplication always dimensionally valid (creates new dimension)
            is_consistent = True
            issue = None

        elif operation == '/':
            # Division always dimensionally valid (creates new dimension)
            is_consistent = True
            issue = None

        else:
            is_consistent = False
            issue = f"Unknown operation: {operation}"

        return DimensionalConsistencyResult(
            expression=f"{left} {operation} {right}",
            is_consistent=is_consistent,
            left_dimension=left_dim.value,
            right_dimension=right_dim.value,
            issue=issue
        )

    def find_dimensionless_groups(self) -> List[Dict[str, Any]]:
        """
        Find dimensionless combinations (Pi groups).

        Dimensionless groups are important because:
        - They are universal (independent of units)
        - They often have physical significance
        - They can reveal scaling laws
        """
        groups = []

        # Simple combinations
        quantities = list(self.quantities.values())

        for i, q1 in enumerate(quantities):
            for j, q2 in enumerate(quantities):
                if i >= j:
                    continue

                # Check if ratio is dimensionless
                if q1.dimension == q2.dimension:
                    groups.append({
                        'name': f"{q1.name}/{q2.name}",
                        'expression': f"{q1.name}/{q2.name}",
                        'type': 'ratio',
                        'significance': 'Dimensionless ratio of like quantities'
                    })

        # Well-known financial dimensionless numbers
        known_groups = [
            {
                'name': 'Sharpe Ratio',
                'expression': 'excess_return / volatility',
                'type': 'risk-adjusted return',
                'significance': 'Return per unit risk'
            },
            {
                'name': 'Coefficient of Variation',
                'expression': 'std / mean',
                'type': 'relative dispersion',
                'significance': 'Normalized measure of variability'
            },
            {
                'name': 'Hurst Exponent',
                'expression': 'log(R/S) / log(n)',
                'type': 'scaling exponent',
                'significance': 'Long-term memory measure'
            }
        ]

        groups.extend(known_groups)

        return groups

    def derive_scaling_laws(self) -> List[str]:
        """
        Derive potential scaling laws from dimensional analysis.

        Scaling laws follow from dimensional consistency.
        """
        laws = []

        # Standard financial scaling laws
        laws.append("Volatility scales as sqrt(time): σ_T = σ_1 × √T")
        laws.append("VaR scales with volatility: VaR_α = z_α × σ × √T")
        laws.append("Sharpe ratio is time-independent for constant parameters")
        laws.append("R/S statistic scales as n^H (Hurst exponent)")
        laws.append("Variance of sum scales with n (independent) or n² (perfectly correlated)")

        return laws

    def analyze(self, quantities_dict: Dict[str, Dict[str, Any]] = None) -> DimensionalReport:
        """
        Complete dimensional analysis.

        Args:
            quantities_dict: Dictionary of {name: {'value': v, 'dimension': d, 'unit': u}}

        Returns:
            DimensionalReport
        """
        if quantities_dict:
            for name, spec in quantities_dict.items():
                self.assign_dimension(
                    name,
                    spec.get('value', 0),
                    spec.get('dimension', 'ratio'),
                    spec.get('unit', '')
                )

        # Run consistency checks on all pairs
        consistency_checks = []
        names = list(self.quantities.keys())

        for i, left in enumerate(names):
            for right in names[i+1:]:
                check = self.check_consistency(left, right, '+')
                consistency_checks.append(check)

        # Find dimensionless groups
        dimensionless_groups = self.find_dimensionless_groups()

        # Derive scaling laws
        scaling_laws = self.derive_scaling_laws()

        # Identify issues
        issues = []
        for check in consistency_checks:
            if check.issue:
                issues.append(check.issue)

        # Check for missing dimensions
        for name, qty in self.quantities.items():
            if qty.dimension == Dimension.RATIO and qty.unit == "":
                issues.append(f"Quantity '{name}' may need explicit dimension assignment")

        return DimensionalReport(
            quantities=list(self.quantities.values()),
            consistency_checks=consistency_checks,
            dimensionless_groups=dimensionless_groups,
            scaling_laws=scaling_laws,
            issues=list(set(issues))  # Deduplicate
        )


def print_dimensional_report(report: DimensionalReport):
    """Pretty-print dimensional report."""
    print("=" * 60)
    print("DIMENSIONAL ANALYSIS")
    print("=" * 60)
    print()

    print("QUANTITIES:")
    print("-" * 40)
    for qty in report.quantities:
        print(f"  {qty.name}: {qty.value} [{qty.dimension.value}] {qty.unit}")

    print()
    print("DIMENSIONLESS GROUPS:")
    print("-" * 40)
    for group in report.dimensionless_groups[:5]:
        print(f"  {group['name']}: {group['expression']}")
        print(f"    ({group['significance']})")

    print()
    print("SCALING LAWS:")
    print("-" * 40)
    for law in report.scaling_laws:
        print(f"  - {law}")

    if report.issues:
        print()
        print("ISSUES:")
        print("-" * 40)
        for issue in report.issues:
            print(f"  ! {issue}")

    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Dimensional Analysis - Demo")
    print("=" * 60)

    analyzer = DimensionalAnalyzer()

    # Define quantities
    quantities = {
        'spy_price': {'value': 450.0, 'dimension': 'price', 'unit': 'USD'},
        'spy_return': {'value': 0.001, 'dimension': 'return', 'unit': ''},
        'spy_volatility': {'value': 0.15, 'dimension': 'volatility', 'unit': 'annual'},
        'risk_free_rate': {'value': 0.05, 'dimension': 'rate', 'unit': 'annual'},
        'vix': {'value': 18.0, 'dimension': 'percent', 'unit': '%'},
        'hurst': {'value': 0.65, 'dimension': 'ratio', 'unit': ''},
        'coherence': {'value': 0.72, 'dimension': 'ratio', 'unit': ''},
        'time_horizon': {'value': 252, 'dimension': 'time', 'unit': 'days'},
    }

    print("\nAnalyzing dimensional consistency...")
    report = analyzer.analyze(quantities)
    print_dimensional_report(report)

    print("\nConsistency checks:")
    for check in report.consistency_checks[:5]:
        status = "OK" if check.is_consistent else "FAIL"
        print(f"  [{status}] {check.expression}: {check.left_dimension} vs {check.right_dimension}")

    print("\nTest completed!")
