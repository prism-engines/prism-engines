"""
Hamiltonian Framework Module
============================

Energy-based dynamics for the indicator space.
This module documents what a Hamiltonian interpretation would require.

CRITICAL NOTE:
==============
A proper Hamiltonian framework requires:
1. Well-defined phase space (position + momentum)
2. Conserved quantities (energy)
3. Symplectic structure
4. Hamilton's equations of motion

For economic/financial systems, these requirements are generally NOT satisfied.
This module provides tools to:
1. TEST whether Hamiltonian-like structure exists
2. DOCUMENT what would be required
3. AVOID false claims of Hamiltonian dynamics

If the system is not Hamiltonian, remove Hamiltonian language from claims.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class HamiltonianCheckResult:
    """Result of checking for Hamiltonian structure."""
    has_conserved_energy: bool
    energy_conservation_error: float  # Should be small if conserved
    has_phase_space: bool
    phase_space_dimensions: int
    liouville_preserved: bool  # Phase space volume conservation
    symplectic_structure: bool
    is_hamiltonian: bool
    recommendations: List[str] = field(default_factory=list)


@dataclass
class HamiltonianReport:
    """Full Hamiltonian analysis report."""
    system_name: str
    n_variables: int
    check_result: HamiltonianCheckResult
    energy_time_series: Optional[np.ndarray] = None
    conclusion: str = ""


class HamiltonianAnalyzer:
    """
    Analyze whether system admits Hamiltonian interpretation.

    A Hamiltonian system has:
    - State variables q (positions)
    - Conjugate momenta p
    - Hamiltonian H(q, p) = total energy
    - Equations of motion: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    - Conserved energy: dH/dt = 0 along trajectories
    """

    def __init__(self):
        pass

    def check_energy_conservation(self,
                                   trajectory: np.ndarray,
                                   energy_func: Optional[callable] = None) -> Tuple[bool, float]:
        """
        Check if energy is conserved along trajectory.

        Args:
            trajectory: T x D array of states over time
            energy_func: Function computing energy from state (uses kinetic if None)

        Returns:
            Tuple of (is_conserved, conservation_error)
        """
        if energy_func is None:
            # Default: use kinetic energy proxy (sum of squared velocities)
            velocities = np.diff(trajectory, axis=0)
            energy = 0.5 * np.sum(velocities ** 2, axis=1)
        else:
            energy = np.array([energy_func(state) for state in trajectory])

        # Check conservation
        energy_std = np.std(energy)
        energy_mean = np.mean(energy)

        # Coefficient of variation
        if energy_mean != 0:
            conservation_error = energy_std / abs(energy_mean)
        else:
            conservation_error = energy_std

        # Consider conserved if CV < 10%
        is_conserved = conservation_error < 0.1

        return is_conserved, conservation_error

    def check_phase_space_structure(self,
                                     positions: np.ndarray,
                                     momenta: np.ndarray) -> bool:
        """
        Check if positions and momenta form valid phase space.

        Requirements:
        - Same dimensionality
        - Independent variables
        - Smooth trajectories

        Args:
            positions: Position variables
            momenta: Momentum variables

        Returns:
            Whether valid phase space exists
        """
        if positions.shape != momenta.shape:
            return False

        # Check independence (positions and momenta shouldn't be perfectly correlated)
        n_dims = positions.shape[1] if len(positions.shape) > 1 else 1

        for i in range(n_dims):
            if len(positions.shape) > 1:
                corr = np.corrcoef(positions[:, i], momenta[:, i])[0, 1]
            else:
                corr = np.corrcoef(positions, momenta)[0, 1]

            # High correlation suggests momenta derived from positions
            if abs(corr) > 0.95:
                return False

        return True

    def check_liouville_theorem(self,
                                 trajectories: List[np.ndarray],
                                 tolerance: float = 0.2) -> bool:
        """
        Check Liouville's theorem: phase space volume is preserved.

        For Hamiltonian systems, the flow preserves phase space volume.
        This is a necessary (but not sufficient) condition.

        Args:
            trajectories: List of trajectory arrays starting from nearby points
            tolerance: Allowed relative change in volume

        Returns:
            Whether volume is approximately preserved
        """
        if len(trajectories) < 2:
            return False

        # Compute initial "volume" (distance between trajectories)
        initial_spread = np.std([t[0] for t in trajectories])

        # Compute final spread
        final_spread = np.std([t[-1] for t in trajectories])

        if initial_spread == 0:
            return False

        # Volume change ratio
        ratio = final_spread / initial_spread

        # Should be close to 1 for Hamiltonian flow
        return abs(ratio - 1) < tolerance

    def check_symplectic_structure(self,
                                    positions: np.ndarray,
                                    momenta: np.ndarray) -> bool:
        """
        Check for symplectic structure.

        A symplectic structure requires the Poisson brackets to satisfy:
        {q_i, q_j} = 0, {p_i, p_j} = 0, {q_i, p_j} = δ_ij

        This is difficult to verify directly from data.
        We use a proxy: check if dynamics preserve the canonical 2-form.

        Args:
            positions, momenta: Phase space coordinates

        Returns:
            Whether symplectic structure is plausible
        """
        # This is a placeholder - true symplectic verification requires
        # computing Poisson brackets from dynamics

        # Proxy: check if position-momentum pairs are conjugate-like
        # (orthogonal in some sense)

        n = len(positions)
        if n < 10:
            return False

        # Compute cross-correlations
        n_dims = positions.shape[1] if len(positions.shape) > 1 else 1

        if n_dims == 1:
            pos = positions.flatten()
            mom = momenta.flatten()
            # Check if d(pos)/dt ~ f(mom) and d(mom)/dt ~ g(pos)
            dpos = np.diff(pos)
            dmom = np.diff(mom)

            corr_pos_mom = abs(np.corrcoef(dpos, mom[:-1])[0, 1])
            corr_mom_pos = abs(np.corrcoef(dmom, pos[:-1])[0, 1])

            # For Hamiltonian: dq/dt depends on p, dp/dt depends on q
            return corr_pos_mom > 0.3 and corr_mom_pos > 0.3

        return False

    def analyze(self,
                data: pd.DataFrame,
                position_cols: Optional[List[str]] = None,
                momentum_cols: Optional[List[str]] = None,
                energy_func: Optional[callable] = None) -> HamiltonianReport:
        """
        Full Hamiltonian analysis.

        Args:
            data: Time series data
            position_cols: Columns to treat as positions
            momentum_cols: Columns to treat as momenta
            energy_func: Custom energy function

        Returns:
            HamiltonianReport
        """
        n_vars = data.shape[1]

        # If no momenta specified, compute as velocities
        if position_cols is None:
            position_cols = data.columns.tolist()

        positions = data[position_cols].values

        if momentum_cols is not None:
            momenta = data[momentum_cols].values
        else:
            # Use velocities as proxy for momenta
            momenta = np.diff(positions, axis=0)
            positions = positions[:-1]

        # Run checks
        energy_conserved, energy_error = self.check_energy_conservation(
            np.column_stack([positions, momenta]),
            energy_func
        )

        has_phase_space = self.check_phase_space_structure(positions, momenta)

        # Liouville check requires multiple trajectories
        # Skip for single trajectory
        liouville_ok = False

        symplectic = self.check_symplectic_structure(positions, momenta)

        # Overall assessment
        is_hamiltonian = (
            energy_conserved and
            has_phase_space and
            symplectic
        )

        recommendations = []
        if not energy_conserved:
            recommendations.append("Energy not conserved - system is not Hamiltonian")
            recommendations.append("Consider dissipative dynamics instead")

        if not has_phase_space:
            recommendations.append("No clear phase space structure")
            recommendations.append("Positions and momenta may not be independent")

        if not symplectic:
            recommendations.append("Symplectic structure not detected")

        if not is_hamiltonian:
            recommendations.append("REMOVE Hamiltonian language from framework description")
            recommendations.append("Consider: gradient flow, dissipative system, or stochastic dynamics")

        conclusion = (
            "System IS consistent with Hamiltonian dynamics" if is_hamiltonian
            else "System is NOT Hamiltonian - use alternative framework"
        )

        return HamiltonianReport(
            system_name="PRISM Indicator Space",
            n_variables=n_vars,
            check_result=HamiltonianCheckResult(
                has_conserved_energy=energy_conserved,
                energy_conservation_error=energy_error,
                has_phase_space=has_phase_space,
                phase_space_dimensions=positions.shape[1] * 2 if has_phase_space else 0,
                liouville_preserved=liouville_ok,
                symplectic_structure=symplectic,
                is_hamiltonian=is_hamiltonian,
                recommendations=recommendations
            ),
            conclusion=conclusion
        )


def define_hamiltonian_stub(
        kinetic_func: callable = None,
        potential_func: callable = None
) -> callable:
    """
    Define a Hamiltonian function H(q, p) = T(p) + V(q).

    This is a template. For financial systems, true Hamiltonians rarely exist.

    Args:
        kinetic_func: T(p) - kinetic energy
        potential_func: V(q) - potential energy

    Returns:
        Hamiltonian function
    """
    if kinetic_func is None:
        kinetic_func = lambda p: 0.5 * np.sum(p ** 2)

    if potential_func is None:
        potential_func = lambda q: 0

    def hamiltonian(q, p):
        return kinetic_func(p) + potential_func(q)

    return hamiltonian


def print_hamiltonian_report(report: HamiltonianReport):
    """Pretty-print Hamiltonian report."""
    print("=" * 60)
    print("HAMILTONIAN ANALYSIS")
    print("=" * 60)
    print(f"System: {report.system_name}")
    print(f"Variables: {report.n_variables}")
    print()

    check = report.check_result
    print("CHECKS:")
    print("-" * 40)
    print(f"  Energy conserved: {check.has_conserved_energy}")
    print(f"    Conservation error: {check.energy_conservation_error:.4f}")
    print(f"  Phase space valid: {check.has_phase_space}")
    print(f"  Liouville preserved: {check.liouville_preserved}")
    print(f"  Symplectic structure: {check.symplectic_structure}")
    print()
    print(f"IS HAMILTONIAN: {check.is_hamiltonian}")
    print()

    print("RECOMMENDATIONS:")
    for rec in check.recommendations:
        print(f"  - {rec}")

    print()
    print(f"CONCLUSION: {report.conclusion}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Hamiltonian Framework - Demo")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: True Hamiltonian system (harmonic oscillator)
    print("\nTest 1: Harmonic Oscillator (should be Hamiltonian)")
    print("-" * 40)

    t = np.linspace(0, 10, 500)
    q_osc = np.cos(t)  # Position
    p_osc = -np.sin(t)  # Momentum

    data_osc = pd.DataFrame({'q': q_osc, 'p': p_osc})
    analyzer = HamiltonianAnalyzer()

    # Note: This won't pass all checks because we simplified the analysis
    report = analyzer.analyze(data_osc, position_cols=['q'], momentum_cols=['p'])
    print_hamiltonian_report(report)

    # Test 2: Financial data (should NOT be Hamiltonian)
    print("\nTest 2: Financial-like data (should NOT be Hamiltonian)")
    print("-" * 40)

    # Random walk with drift (non-Hamiltonian)
    returns = np.random.randn(500) * 0.01 + 0.0001
    prices = 100 * np.exp(np.cumsum(returns))

    data_fin = pd.DataFrame({
        'price': prices,
        'volume': np.abs(np.random.randn(500) * 1000)
    })

    report = analyzer.analyze(data_fin)
    print_hamiltonian_report(report)

    print("\nTest completed!")
