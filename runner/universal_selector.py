"""
Universal Input Selector for PRISM
==================================

Allows selection of ANY indicators from ANY domain for cross-domain analysis.
No artificial boundaries between economic, climate, biological, social data.

The core VCF insight: All systems exhibit mathematical rhythms that can be
compared regardless of their source domain.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class AnalysisMode(Enum):
    """Analysis approach selector."""
    ML = "ml"           # Machine learning / pattern recognition
    META = "meta"       # Pure mathematical / deterministic
    HYBRID = "hybrid"   # Both approaches


@dataclass
class IndicatorSpec:
    """Specification for a single indicator."""
    id: str
    name: str
    domain: str
    description: str
    source: str
    frequency: str
    data_type: str
    transform: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_id(self) -> str:
        """Domain-qualified identifier."""
        return f"{self.domain}.{self.id}"


@dataclass
class SelectionSet:
    """A user's selected indicators for analysis."""
    indicators: List[IndicatorSpec]
    mode: AnalysisMode
    engines: List[str]
    name: Optional[str] = None

    @property
    def domains(self) -> Set[str]:
        """Unique domains in selection."""
        return {ind.domain for ind in self.indicators}

    @property
    def is_cross_domain(self) -> bool:
        """True if selection spans multiple domains."""
        return len(self.domains) > 1

    def summary(self) -> Dict[str, Any]:
        """Selection summary for display."""
        return {
            'total_indicators': len(self.indicators),
            'domains': list(self.domains),
            'is_cross_domain': self.is_cross_domain,
            'mode': self.mode.value,
            'engines': self.engines,
            'by_domain': {
                domain: [i.id for i in self.indicators if i.domain == domain]
                for domain in self.domains
            }
        }


class UniversalSelector:
    """
    Universal indicator selector for cross-domain analysis.

    Loads the universal registry and allows users to select any
    combination of indicators from any domains.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize selector with registry.

        Args:
            registry_path: Path to universal_registry.yaml
        """
        if registry_path is None:
            # Try multiple possible locations
            possible_paths = [
                Path("data/universal_registry.yaml"),
                Path(__file__).parent.parent / "data" / "universal_registry.yaml",
            ]
            for p in possible_paths:
                if p.exists():
                    registry_path = p
                    break
            else:
                registry_path = possible_paths[0]  # Default to first option

        self.registry_path = registry_path
        self.registry: Dict[str, Any] = {}
        self.indicators: Dict[str, IndicatorSpec] = {}
        self._load_registry()

    def _load_registry(self):
        """Load and index the universal registry."""
        if not self.registry_path.exists():
            # Create default registry structure
            self.registry = {'domains': {}, 'version': '1.0.0'}
            return

        with open(self.registry_path) as f:
            self.registry = yaml.safe_load(f)

        # Index all indicators
        for domain_id, domain_data in self.registry.get('domains', {}).items():
            for ind_id, ind_data in domain_data.get('indicators', {}).items():
                spec = IndicatorSpec(
                    id=ind_id,
                    name=ind_data.get('name', ind_id),
                    domain=domain_id,
                    description=ind_data.get('description', ''),
                    source=ind_data.get('source', 'unknown'),
                    frequency=ind_data.get('frequency', 'daily'),
                    data_type=ind_data.get('data_type', 'level'),
                    transform=ind_data.get('transform', 'normalize'),
                    metadata=ind_data
                )
                self.indicators[spec.full_id] = spec

    def list_domains(self) -> List[Dict[str, Any]]:
        """List all available domains."""
        domains = []
        for domain_id, domain_data in self.registry.get('domains', {}).items():
            domains.append({
                'id': domain_id,
                'name': domain_id.replace('_', ' ').title(),
                'description': domain_data.get('description', ''),
                'color': domain_data.get('color', '#666666'),
                'indicator_count': len(domain_data.get('indicators', {}))
            })
        return domains

    def list_indicators(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available indicators, optionally filtered by domain.

        Args:
            domain: Optional domain filter

        Returns:
            List of indicator info dicts for UI display
        """
        indicators = []

        for full_id, spec in self.indicators.items():
            if domain and spec.domain != domain:
                continue

            indicators.append({
                'id': spec.id,
                'full_id': full_id,
                'name': spec.name,
                'domain': spec.domain,
                'description': spec.description,
                'frequency': spec.frequency,
                'data_type': spec.data_type
            })

        return sorted(indicators, key=lambda x: (x['domain'], x['name']))

    def get_indicator(self, indicator_id: str) -> Optional[IndicatorSpec]:
        """
        Get indicator specification.

        Args:
            indicator_id: Either short id or full_id (domain.id)

        Returns:
            IndicatorSpec or None
        """
        # Try full_id first
        if indicator_id in self.indicators:
            return self.indicators[indicator_id]

        # Try to find by short id (may be ambiguous)
        matches = [spec for spec in self.indicators.values() if spec.id == indicator_id]
        if len(matches) == 1:
            return matches[0]

        return None

    def create_selection(self,
                        indicator_ids: List[str],
                        mode: str = "meta",
                        engines: Optional[List[str]] = None,
                        name: Optional[str] = None) -> SelectionSet:
        """
        Create a selection set from indicator IDs.

        Args:
            indicator_ids: List of indicator IDs (full or short)
            mode: 'ml', 'meta', or 'hybrid'
            engines: List of engine names to use
            name: Optional name for this selection

        Returns:
            SelectionSet ready for analysis
        """
        selected = []

        for ind_id in indicator_ids:
            spec = self.get_indicator(ind_id)
            if spec:
                selected.append(spec)

        return SelectionSet(
            indicators=selected,
            mode=AnalysisMode(mode),
            engines=engines or ['spectral_coherence', 'hurst_exponent'],
            name=name
        )

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get pre-defined cross-domain hypotheses."""
        return self.registry.get('cross_domain_hypotheses', [])

    def validate_selection(self, selection: SelectionSet) -> Dict[str, Any]:
        """
        Validate a selection for analysis readiness.

        Returns validation status and any issues.
        """
        issues = []
        warnings = []

        if len(selection.indicators) < 2:
            issues.append("Need at least 2 indicators for coherence analysis")

        # Check frequency compatibility
        frequencies = {spec.frequency for spec in selection.indicators}
        if len(frequencies) > 1:
            warnings.append(f"Mixed frequencies: {frequencies}. Will resample to common frequency.")

        # Check data type compatibility
        data_types = {spec.data_type for spec in selection.indicators}
        if 'trending' in data_types and 'oscillating' in data_types:
            warnings.append("Mix of trending and oscillating data. Consider detrending.")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'selection_summary': selection.summary()
        }

    def get_registry_for_ui(self) -> Dict[str, Any]:
        """
        Get registry data formatted for the web UI.

        Returns dict with domains and their indicators ready for JSON serialization.
        """
        ui_data = {
            'domains': {},
            'version': self.registry.get('version', '1.0.0')
        }

        for domain_id, domain_data in self.registry.get('domains', {}).items():
            ui_data['domains'][domain_id] = {
                'name': domain_id.replace('_', ' ').title(),
                'description': domain_data.get('description', ''),
                'color': domain_data.get('color', '#666666'),
                'indicators': []
            }

            for ind_id, ind_data in domain_data.get('indicators', {}).items():
                ui_data['domains'][domain_id]['indicators'].append({
                    'id': ind_id,
                    'full_id': f"{domain_id}.{ind_id}",
                    'name': ind_data.get('name', ind_id),
                    'description': ind_data.get('description', ''),
                    'frequency': ind_data.get('frequency', 'daily'),
                    'data_type': ind_data.get('data_type', 'level')
                })

        return ui_data


# Convenience function for quick selection
def quick_select(*indicator_ids, mode: str = "meta") -> SelectionSet:
    """
    Quick selection helper.

    Usage:
        selection = quick_select("spy_ma_ratio", "enso_index", "flu_activity")
    """
    selector = UniversalSelector()
    return selector.create_selection(list(indicator_ids), mode=mode)


if __name__ == "__main__":
    # Demo the selector
    print("=" * 60)
    print("Universal Selector Demo")
    print("=" * 60)

    selector = UniversalSelector()

    print("\nAvailable Domains:")
    for domain in selector.list_domains():
        print(f"  {domain['id']}: {domain['indicator_count']} indicators")

    print("\nAll Indicators:")
    for ind in selector.list_indicators():
        print(f"  [{ind['domain']}] {ind['id']}: {ind['name']}")

    # Create a cross-domain selection
    selection = selector.create_selection(
        ['economic.spy_ma_ratio', 'climate.enso_index', 'biological.flu_activity'],
        mode='meta',
        engines=['spectral_coherence', 'hurst_exponent']
    )

    print(f"\nCross-Domain Selection:")
    print(f"  Indicators: {len(selection.indicators)}")
    print(f"  Domains: {selection.domains}")
    print(f"  Is Cross-Domain: {selection.is_cross_domain}")

    validation = selector.validate_selection(selection)
    print(f"\nValidation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Warnings: {validation['warnings']}")
