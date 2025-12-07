"""
Indicator Family Manager
========================

Manages multi-resolution and multi-source versions of the same underlying
signal to prevent double-counting in analysis.

Key Concepts:
- Family: A group of indicators representing the same underlying signal
- Member: A specific version (resolution/source) within a family
- Purpose: The analysis type that determines which member to use

Example:
    The S&P 500 exists as:
    - sp500_d (daily from Tiingo, 1928-present) → use for geometry
    - sp500_m (monthly from FRED, 1871-present) → use for calibration

    Without family management, loading both would double-weight SPX.
    With family management, PRISM auto-selects the right one per purpose.

Usage:
    manager = FamilyManager()

    # Get appropriate indicator for purpose
    indicator = manager.get_for_purpose('spx', 'geometry')  # Returns 'sp500_d'
    indicator = manager.get_for_purpose('spx', 'calibration')  # Returns 'sp500_m'

    # Check for duplicates before analysis
    warnings = manager.check_duplicates(data_columns)

    # Validate a selection
    issues = manager.validate_selection(['sp500_d', 'sp500_m', 'vix_d'])

Author: PRISM Project
Version: 1.0.0
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


class Resolution(Enum):
    """Data resolution levels."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    MULTI = "multi"  # Explicit multi-resolution analysis


@dataclass
class FamilyMember:
    """A specific version of an indicator within a family."""
    id: str
    source: str
    resolution: str
    history_start: int
    use_for: List[str]
    symbol: Optional[str] = None
    series: Optional[str] = None
    notes: Optional[str] = None

    @property
    def resolution_enum(self) -> Resolution:
        return Resolution(self.resolution)


@dataclass
class IndicatorFamily:
    """A family of related indicators (same underlying signal)."""
    id: str
    canonical_name: str
    description: str
    members: Dict[str, FamilyMember]
    rules: Dict[str, Any]

    @property
    def default_member(self) -> str:
        """Get default representation."""
        return self.rules.get('default_representation', list(self.members.keys())[0])

    @property
    def allows_multi_resolution(self) -> bool:
        """Whether family allows multiple resolutions in same analysis."""
        return self.rules.get('allow_multi_resolution', False)

    @property
    def correlation_threshold(self) -> float:
        """Threshold for duplicate warning."""
        return self.rules.get('correlation_warning_threshold', 0.90)

    def get_member_for_purpose(self, purpose: str) -> Optional[str]:
        """Get the appropriate member for a given analysis purpose."""
        for member_id, member in self.members.items():
            if purpose in member.use_for:
                return member_id
        return self.default_member

    def get_members_by_resolution(self, resolution: str) -> List[str]:
        """Get all members with a specific resolution."""
        return [
            member_id for member_id, member in self.members.items()
            if member.resolution == resolution
        ]


class FamilyManager:
    """
    Manages indicator families to prevent double-counting.

    Responsibilities:
    1. Load and index family registry
    2. Select appropriate indicator version by purpose
    3. Detect potential duplicates in analysis
    4. Validate indicator selections
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize family manager.

        Args:
            registry_path: Path to families.yaml
        """
        self.registry_path = registry_path or Path(__file__).parent / "registry" / "families.yaml"
        self.families: Dict[str, IndicatorFamily] = {}
        self.member_to_family: Dict[str, str] = {}  # Reverse lookup
        self.purpose_resolution_map: Dict[str, str] = {}
        self.global_rules: Dict[str, Any] = {}

        self._load_registry()

    def _load_registry(self):
        """Load family registry from YAML."""
        if not self.registry_path.exists():
            warnings.warn(f"Family registry not found at {self.registry_path}")
            return

        with open(self.registry_path) as f:
            data = yaml.safe_load(f)

        # Load global settings
        self.purpose_resolution_map = data.get('purpose_resolution_map', {})
        self.global_rules = data.get('global_rules', {})

        # Load families
        for family_id, family_data in data.get('families', {}).items():
            members = {}
            for member_id, member_data in family_data.get('members', {}).items():
                members[member_id] = FamilyMember(
                    id=member_id,
                    source=member_data.get('source', 'unknown'),
                    resolution=member_data.get('resolution', 'daily'),
                    history_start=member_data.get('history_start', 2000),
                    use_for=member_data.get('use_for', []),
                    symbol=member_data.get('symbol'),
                    series=member_data.get('series'),
                    notes=member_data.get('notes')
                )
                # Build reverse lookup
                self.member_to_family[member_id] = family_id

            self.families[family_id] = IndicatorFamily(
                id=family_id,
                canonical_name=family_data.get('canonical_name', family_id),
                description=family_data.get('description', ''),
                members=members,
                rules=family_data.get('rules', {})
            )

    def get_family(self, family_id: str) -> Optional[IndicatorFamily]:
        """Get a family by ID."""
        return self.families.get(family_id)

    def get_family_for_member(self, member_id: str) -> Optional[IndicatorFamily]:
        """Get the family that contains a specific member."""
        family_id = self.member_to_family.get(member_id)
        if family_id:
            return self.families.get(family_id)
        return None

    def get_for_purpose(self, family_id: str, purpose: str) -> Optional[str]:
        """
        Get the appropriate indicator member for a given purpose.

        Args:
            family_id: Family identifier (e.g., 'spx')
            purpose: Analysis purpose (e.g., 'geometry', 'calibration')

        Returns:
            Member ID or None if family not found
        """
        family = self.families.get(family_id)
        if not family:
            return None

        return family.get_member_for_purpose(purpose)

    def select_indicators(self,
                          requested: List[str],
                          purpose: str) -> Tuple[List[str], List[str]]:
        """
        Select appropriate indicators for a purpose, resolving families.

        Args:
            requested: List of requested indicator IDs (can be family IDs or member IDs)
            purpose: Analysis purpose

        Returns:
            Tuple of (selected_indicators, warnings)
        """
        selected = []
        warns = []
        seen_families = set()

        for req in requested:
            # Check if it's a family ID
            if req in self.families:
                family = self.families[req]
                member_id = family.get_member_for_purpose(purpose)

                if family.id in seen_families:
                    warns.append(f"Family '{family.id}' already represented - skipping {req}")
                    continue

                selected.append(member_id)
                seen_families.add(family.id)

            # Check if it's a member ID
            elif req in self.member_to_family:
                family_id = self.member_to_family[req]

                if family_id in seen_families:
                    warns.append(
                        f"Duplicate family signal: '{req}' belongs to family '{family_id}' "
                        f"which is already represented"
                    )
                    continue

                selected.append(req)
                seen_families.add(family_id)

            else:
                # Not in registry - pass through as-is
                selected.append(req)

        return selected, warns

    def check_duplicates(self,
                         indicators: List[str],
                         data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Check for potential duplicate signals in indicator list.

        Args:
            indicators: List of indicator IDs
            data: Optional DataFrame for correlation-based detection

        Returns:
            List of warning dicts with details
        """
        warnings_list = []

        # Check 1: Multiple members from same family
        family_counts: Dict[str, List[str]] = {}
        for ind in indicators:
            family_id = self.member_to_family.get(ind)
            if family_id:
                if family_id not in family_counts:
                    family_counts[family_id] = []
                family_counts[family_id].append(ind)

        for family_id, members in family_counts.items():
            if len(members) > 1:
                family = self.families[family_id]
                if not family.allows_multi_resolution:
                    warnings_list.append({
                        'type': 'family_duplicate',
                        'severity': 'high',
                        'family': family_id,
                        'members': members,
                        'message': (
                            f"Multiple members of family '{family.canonical_name}' loaded: "
                            f"{members}. This may double-weight the signal."
                        ),
                        'suggestion': f"Use only one: recommend '{family.default_member}'"
                    })
                else:
                    warnings_list.append({
                        'type': 'family_multi_resolution',
                        'severity': 'info',
                        'family': family_id,
                        'members': members,
                        'message': (
                            f"Multi-resolution loading for '{family.canonical_name}': {members}. "
                            f"This is allowed for this family."
                        )
                    })

        # Check 2: Correlation-based detection (if data provided)
        if data is not None:
            threshold = self.global_rules.get('duplicate_correlation_threshold', 0.90)
            corr_warnings = self._check_correlation_duplicates(data, indicators, threshold)
            warnings_list.extend(corr_warnings)

        return warnings_list

    def _check_correlation_duplicates(self,
                                       data: pd.DataFrame,
                                       indicators: List[str],
                                       threshold: float) -> List[Dict[str, Any]]:
        """Check for high-correlation pairs that might be duplicates."""
        warnings_list = []

        available = [ind for ind in indicators if ind in data.columns]

        for i, ind1 in enumerate(available):
            for ind2 in available[i+1:]:
                # Compute correlation at common frequency
                series1 = data[ind1].dropna()
                series2 = data[ind2].dropna()

                # Align series
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 30:
                    continue

                corr = np.corrcoef(
                    series1.loc[common_idx].values,
                    series2.loc[common_idx].values
                )[0, 1]

                if abs(corr) > threshold:
                    # Check if they're from same family (already warned)
                    fam1 = self.member_to_family.get(ind1)
                    fam2 = self.member_to_family.get(ind2)

                    if fam1 and fam1 == fam2:
                        continue  # Already warned via family check

                    warnings_list.append({
                        'type': 'correlation_duplicate',
                        'severity': 'medium',
                        'indicators': [ind1, ind2],
                        'correlation': round(corr, 4),
                        'message': (
                            f"High correlation ({corr:.2f}) between '{ind1}' and '{ind2}'. "
                            f"These may represent the same underlying signal."
                        ),
                        'suggestion': "Consider removing one or verify they measure different phenomena"
                    })

        return warnings_list

    def validate_selection(self,
                           indicators: List[str],
                           purpose: str = 'geometry') -> Dict[str, Any]:
        """
        Validate an indicator selection for analysis.

        Args:
            indicators: List of indicator IDs
            purpose: Analysis purpose

        Returns:
            Validation result with status, issues, and suggestions
        """
        issues = []
        suggestions = []

        # Check for duplicates
        dup_warnings = self.check_duplicates(indicators)
        high_severity = [w for w in dup_warnings if w['severity'] == 'high']

        if high_severity:
            issues.extend([w['message'] for w in high_severity])
            suggestions.extend([w['suggestion'] for w in high_severity])

        # Check resolution consistency
        resolutions = set()
        for ind in indicators:
            family = self.get_family_for_member(ind)
            if family and ind in family.members:
                resolutions.add(family.members[ind].resolution)

        if len(resolutions) > 1 and purpose != 'wavelet':
            issues.append(
                f"Mixed resolutions in selection: {resolutions}. "
                f"For '{purpose}' analysis, use consistent resolution."
            )
            preferred_res = self.purpose_resolution_map.get(purpose, 'daily')
            suggestions.append(f"Recommended resolution for '{purpose}': {preferred_res}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'warnings': dup_warnings,
            'indicator_count': len(indicators)
        }

    def list_families(self) -> List[Dict[str, Any]]:
        """List all registered families."""
        return [
            {
                'id': fam.id,
                'name': fam.canonical_name,
                'description': fam.description,
                'members': list(fam.members.keys()),
                'default': fam.default_member,
                'allows_multi_resolution': fam.allows_multi_resolution
            }
            for fam in self.families.values()
        ]

    def get_resolution_for_purpose(self, purpose: str) -> str:
        """Get recommended resolution for a purpose."""
        return self.purpose_resolution_map.get(purpose, 'daily')


# =============================================================================
# Pre-Analysis Validation Hook
# =============================================================================

def validate_before_analysis(indicators: List[str],
                              data: Optional[pd.DataFrame] = None,
                              purpose: str = 'geometry',
                              strict: bool = False) -> Tuple[List[str], bool]:
    """
    Validate and filter indicators before analysis.

    This function should be called before any analysis to:
    1. Detect duplicate signals
    2. Filter to appropriate resolution
    3. Warn about potential issues

    Args:
        indicators: Requested indicator list
        data: Optional data for correlation checking
        purpose: Analysis purpose
        strict: If True, raise exception on issues

    Returns:
        Tuple of (filtered_indicators, is_valid)
    """
    manager = FamilyManager()

    # Select appropriate indicators
    selected, selection_warnings = manager.select_indicators(indicators, purpose)

    # Validate
    validation = manager.validate_selection(selected, purpose)

    # Print warnings
    for warn in selection_warnings:
        print(f"[WARNING] {warn}")

    for warn in validation['warnings']:
        severity_icon = {'high': '[ERROR]', 'medium': '[WARNING]', 'info': '[INFO]'}
        icon = severity_icon.get(warn['severity'], '[NOTE]')
        print(f"{icon} {warn['message']}")

    if validation['issues']:
        for issue in validation['issues']:
            print(f"[ISSUE] {issue}")

        if strict:
            raise ValueError(f"Validation failed: {validation['issues']}")

    # Check correlations if data provided
    if data is not None:
        corr_warnings = manager.check_duplicates(selected, data)
        for warn in corr_warnings:
            if warn['type'] == 'correlation_duplicate':
                print(f"[WARNING] {warn['message']}")

    return selected, validation['valid']


# =============================================================================
# CLI / Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Indicator Family Manager - Demo")
    print("=" * 60)

    manager = FamilyManager()

    print("\n[FAMILIES] Registered Families:")
    for fam in manager.list_families():
        print(f"\n  {fam['id']} ({fam['name']})")
        print(f"    Members: {fam['members']}")
        print(f"    Default: {fam['default']}")
        print(f"    Multi-res: {fam['allows_multi_resolution']}")

    print("\n" + "-" * 60)
    print("[PURPOSE] Purpose-Based Selection:")

    # Test purpose-based selection
    for purpose in ['geometry', 'calibration', 'wavelet']:
        spx = manager.get_for_purpose('spx', purpose)
        print(f"  SPX for '{purpose}': {spx}")

    print("\n" + "-" * 60)
    print("[DUPLICATES] Duplicate Detection Test:")

    # Test duplicate detection
    test_indicators = ['sp500_d', 'sp500_m', 'vix_d', 'dgs10_d']
    warnings_list = manager.check_duplicates(test_indicators)

    for warn in warnings_list:
        print(f"  [{warn['severity'].upper()}] {warn['message']}")

    print("\n" + "-" * 60)
    print("[VALIDATION] Validation Test:")

    # Test validation
    validation = manager.validate_selection(test_indicators, purpose='geometry')
    print(f"  Valid: {validation['valid']}")
    print(f"  Issues: {validation['issues']}")
    print(f"  Suggestions: {validation['suggestions']}")

    print("\n" + "-" * 60)
    print("[SELECTION] Smart Selection Test:")

    # Test smart selection (using family IDs)
    requested = ['spx', 'treasury_10y', 'vix', 'usd_index']
    selected, warns = manager.select_indicators(requested, purpose='geometry')

    print(f"  Requested: {requested}")
    print(f"  Selected:  {selected}")
    if warns:
        print(f"  Warnings:  {warns}")

    print("\n" + "=" * 60)
    print("Demo completed!")
