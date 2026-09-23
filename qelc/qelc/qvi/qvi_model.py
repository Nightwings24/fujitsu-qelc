"""Quantum Viability Index (QVI) with real, cited inputs.

The proposal's QVI = security_bits^2 / (alpha * gate_cost + beta * latency)
was previously filled with a qualitative Low/Medium/High table and framed as
"benchmarking Dilithium on the quantum simulator" — which is not meaningful
(Dilithium is a classical signature scheme). Here the same index is grounded:

  * security side: NIST security-category equivalent bits, anchored by
    published quantum attack-cost estimates for the matching AES Grover
    key-search instance (Grassl et al. 2016, "Applying Grover's algorithm to
    AES", PQCrypto; Jaques et al. 2020, EUROCRYPT). Our own Grover runs
    (qelc.qvi.grover) empirically verify the sqrt(2^k) iteration scaling that
    those estimates extrapolate.
  * cost side: the *classical operational* cost banks actually pay — signing/
    verification time and bandwidth. Defaults below are the round-3 Dilithium
    reference-implementation numbers (Skylake cycle counts) from the NIST
    submission package; replace with team-measured values (e.g. liboqs) for
    the final report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PQCLevel:
    name: str
    nist_category: int
    security_bits: int              # classical-equivalent target
    attack_log2_gates: float        # literature quantum attack cost (log2 gates)
    sign_kcycles: float             # reference sign cost (kilocycles)
    verify_kcycles: float           # reference verify cost (kilocycles)
    signature_bytes: int
    public_key_bytes: int


# Literature reference values; see module docstring for sources.
# attack_log2_gates: Grassl et al. 2016 T-gate counts for Grover on
# AES-128/192/256 (~2^86, ~2^118, ~2^151), the standard anchors for NIST
# categories 1/3/5. Dilithium2 is formally category 2; its exhaustive-search
# anchor is treated as the 128-bit level for comparability with the proposal.
DILITHIUM_LEVELS = [
    PQCLevel("Dilithium2", 2, 128, 86.0, 333.0, 118.0, 2420, 1312),
    PQCLevel("Dilithium3", 3, 192, 118.0, 529.0, 179.0, 3293, 1952),
    PQCLevel("Dilithium5", 5, 256, 151.0, 642.0, 279.0, 4595, 2592),
]


def compute_qvi(
    levels: list = DILITHIUM_LEVELS,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    required_bits: int = 192,
):
    """QVI table with two variants.

    qvi_raw is the proposal's original formula, bits^2 / cost. Fed with real
    cost data it selects the HIGHEST level: Dilithium's operational costs grow
    roughly linearly with level (not "exponentially" as the proposal claimed),
    so a squared reward always outruns them. That falsifies the original
    "diminishing returns" story — worth stating openly in the report.

    qvi_anchored fixes the economics: security utility saturates at the
    mandated compliance floor (min(bits, required)^2) because bits beyond the
    requirement have bounded marginal value to a bank. Among compliant levels
    this selects the cheapest one — Dilithium3 at the 192-bit floor.

    Costs are normalized to the weakest level so alpha (compute) and beta
    (bandwidth) weigh dimensionless quantities.
    """
    import pandas as pd

    base_ops = min(l.sign_kcycles + l.verify_kcycles for l in levels)
    base_bw = min(l.signature_bytes + l.public_key_bytes for l in levels)
    rows = []
    for l in levels:
        ops = (l.sign_kcycles + l.verify_kcycles) / base_ops
        bw = (l.signature_bytes + l.public_key_bytes) / base_bw
        denom = alpha * ops + beta * bw
        rows.append(
            {
                "level": l.name,
                "nist_category": l.nist_category,
                "security_bits": l.security_bits,
                "attack_log2_gates": l.attack_log2_gates,
                "rel_compute_cost": round(ops, 3),
                "rel_bandwidth_cost": round(bw, 3),
                "qvi_raw": round(l.security_bits**2 / denom, 1),
                "qvi_anchored": round(min(l.security_bits, required_bits) ** 2 / denom, 1),
                "meets_requirement": l.security_bits >= required_bits,
            }
        )
    df = pd.DataFrame(rows)
    eligible = df[df["meets_requirement"]]
    df["recommended"] = df["level"] == (
        eligible.loc[eligible["qvi_anchored"].idxmax(), "level"] if not eligible.empty else None
    )
    return df
