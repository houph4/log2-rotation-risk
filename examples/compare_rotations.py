"""Compare a proposed rotation with the current estimate under a local bank."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerical_release"))
from geometry import allowance, contrasts, decode_moments, encode_moments, polynomial


def main():
    rotations = np.array([[0.02, 0, 0.10], [-0.02, 0, 0.20], [0, 0.02, 0.15]])
    weights = np.array([0.2, 0.3, 0.5])
    proposal = np.array([0, 0, 0.15])
    current = np.zeros(3)
    support_radius = 0.25  # radians

    message = encode_moments(rotations, weights)
    mean, second_moment = decode_moments(message)
    score = polynomial(proposal, current, mean, second_moment)[0]
    error = allowance(proposal, current, support_radius, message)[0]
    exact = contrasts(proposal, current, rotations, weights)[0]
    if score + error < 0:
        decision = "accept"
    elif score - error > 0:
        decision = "retain"
    else:
        decision = "unresolved"
    assert len(message) == 36
    assert score - error <= exact <= score + error
    print(f"Log2 message: {len(message)} bytes")
    print(f"Log2 risk difference: {score:.9f} rad^2")
    print(f"Certified interval: [{score-error:.9f}, {score+error:.9f}] rad^2")
    print(f"FullPool risk difference: {exact:.9f} rad^2")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
