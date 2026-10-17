#!/usr/bin/env python3
"""Exact finite-product checks for a boundary-sensitive deep-ReLU expectation bound.

These constructed networks are mathematical experiments, not trained models or performance
benchmarks. A summary is built before any output readout is supplied. Every numerical
calculation uses fractions; the 64-input references exploit explicit affine or permutation
structure rather than enumerate 2**64 inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from fractions import Fraction as Q
from pathlib import Path


def dot(left, right):
    assert len(left) == len(right)
    return sum((x * y for x, y in zip(left, right)), Q(0))


def forward(layers, values):
    trace = []
    for matrix, bias in layers:
        pre = [dot(row, values) + offset for row, offset in zip(matrix, bias)]
        values = [max(Q(0), value) for value in pre]
        trace.append({"pre": pre, "output": values})
    return values, trace


def signs(count):
    return itertools.product((Q(-1), Q(1)), repeat=count)


def prepare(layers, width, kept):
    """No readout argument: retain the first input blocks and center the other signs."""
    cells = []
    for retained in signs(kept):
        values = list(retained) + [Q(0)] * (width - kept)
        radii = [Q(0)] * kept + [Q(1)] * (width - kept)
        trace = []
        for matrix, bias in layers:
            pre = [dot(row, values) + offset for row, offset in zip(matrix, bias)]
            delta = [dot([abs(weight) for weight in row], radii) for row in matrix]
            output = [max(Q(0), value) for value in pre]
            trace.append({
                "pre": pre,
                "delta": delta,
                "ramp": [max(Q(0), radius - abs(value)) for value, radius in zip(pre, delta)],
                "gates": [value > 0 for value in pre],
                "output": output,
            })
            # The next affine map needs an enclosure for the full activation error.
            radii = [
                max(abs(max(Q(0), value - radius) - center),
                    abs(max(Q(0), value + radius) - center))
                for value, radius, center in zip(pre, delta, output)
            ]
            values = output
        cells.append({"retained": retained, "trace": trace, "output": values})
    return {"input_width": width, "retained_inputs": kept, "cells": cells}


def adjoints(layers, trace, readout):
    """Suffix weights use the fixed surrogate gates, including their actual signs."""
    suffix = list(readout)
    result = [None] * len(layers)
    for index in reversed(range(len(layers))):
        result[index] = suffix
        matrix, _ = layers[index]
        suffix = [
            sum((matrix[i][j] * suffix[i] * trace[index]["gates"][i]
                 for i in range(len(matrix))), Q(0))
            for j in range(len(matrix[0]))
        ]
    return result, suffix


def estimate(layers, summary, readout, bias):
    prediction = bound = lower = upper = uniform = Q(0)
    for cell in summary["cells"]:
        prediction += dot(readout, cell["output"]) + bias
        suffixes, _ = adjoints(layers, cell["trace"], readout)
        for suffix, layer in zip(suffixes, cell["trace"]):
            bound += dot([abs(value) for value in suffix], layer["ramp"])
            uniform += dot([abs(value) for value in suffix], layer["delta"])
            lower += dot([min(Q(0), value) for value in suffix], layer["ramp"])
            upper += dot([max(Q(0), value) for value in suffix], layer["ramp"])
    count = len(summary["cells"])
    return prediction / count, bound / count, lower / count, upper / count, uniform / count


def network(width, stable=False):
    """Four retained signs and dense, nonzero discarded columns in the first layer."""
    assert width > 4
    discarded = width - 4
    if stable:
        first = [
            [Q(1, 4), Q(1, 8), Q(0), Q(0)] + [Q(1, 256)] * discarded,
            [Q(0), Q(0), Q(-1, 8), Q(0)] +
            [Q((-1) ** i, 512) for i in range(discarded)],
            [Q(0), Q(0), Q(0), Q(1, 4)] + [Q(1, 256)] * discarded,
        ]
        return [
            (first, [Q(2), Q(3), Q(-2)]),
            ([[Q(1), Q(-1, 4), Q(1)], [Q(-1), Q(0), Q(0)],
              [Q(0), Q(1, 2), Q(0)]], [Q(1), Q(-2), Q(1)]),
            ([[Q(2), Q(1), Q(1, 3)], [Q(1), Q(0), Q(-1)]], [Q(-1), Q(4)]),
        ]
    # Keep the discarded block's worst-case radius fixed when its dimension grows.
    first = [
        [Q(1, 2), Q(1, 4), Q(0), Q(0)] + [Q(1, 4 * discarded)] * discarded,
        [Q(0), Q(0), Q(-1, 2), Q(0)] + [Q(1, 8 * discarded)] * discarded,
        [Q(0), Q(0), Q(0), Q(1, 4)] + [Q(-1, 4 * discarded)] * discarded,
    ]
    return [
        (first, [Q(-1, 8), Q(1, 4), Q(-1, 4)]),
        ([[Q(1), Q(-1, 2), Q(1, 4)], [Q(-1, 2), Q(1), Q(1, 2)]],
         [Q(0), Q(1, 8)]),
        ([[Q(1), Q(-1, 3)], [Q(1, 4), Q(1)]], [Q(-1, 16), Q(-1, 8)]),
    ]


def affine_reference(layers, width):
    """Independently collapse a globally stable network to an affine map on the full cube."""
    coefficients = [[Q(i == j) for j in range(width)] for i in range(width)]
    constants = [Q(0)] * width
    patterns = []
    for matrix, bias in layers:
        next_constants = [dot(row, constants) + value for row, value in zip(matrix, bias)]
        next_coefficients = [
            [sum((row[k] * coefficients[k][j] for k in range(len(row))), Q(0))
             for j in range(width)]
            for row in matrix
        ]
        active = []
        for i, (center, row) in enumerate(zip(next_constants, next_coefficients)):
            radius = sum(map(abs, row), Q(0))
            if center - radius >= 0:
                active.append(True)
            elif center + radius <= 0:
                active.append(False)
                next_constants[i] = Q(0)
                next_coefficients[i] = [Q(0)] * width
            else:
                raise AssertionError("affine reference encountered an unstable gate")
        constants, coefficients = next_constants, next_coefficients
        patterns.append(active)
    # Every original sign has mean zero, so only the affine constant remains.
    return constants, patterns


def binomial_reference(layers, width, kept):
    """Identical discarded columns make the network depend only on their signed sum."""
    discarded = width - kept
    assert all(len(set(row[kept:])) == 1 for row in layers[0][0])
    mean = [Q(0)] * len(layers[-1][0])
    evaluated = 0
    for retained in signs(kept):
        for positive in range(discarded + 1):
            values = list(retained) + [Q(1)] * positive + [Q(-1)] * (discarded - positive)
            output, _ = forward(layers, values)
            probability = Q(math.comb(discarded, positive), 2 ** width)
            mean = [old + probability * value for old, value in zip(mean, output)]
            evaluated += 1
    return mean, evaluated


def exhaustive_reference(layers, summary, readouts):
    """Check both the finite expectation and the exact layerwise error identity."""
    width, kept = summary["input_width"], summary["retained_inputs"]
    cells = {tuple(cell["retained"]): cell for cell in summary["cells"]}
    mean = [Q(0)] * len(layers[-1][0])
    identities = 0
    for values in signs(width):
        output, trace = forward(layers, values)
        mean = [old + value / (2 ** width) for old, value in zip(mean, output)]
        cell = cells[values[:kept]]
        residual = [Q(0)] * kept + list(values[kept:])
        remainders = []
        for actual, center in zip(trace, cell["trace"]):
            difference = [x - y for x, y in zip(actual["pre"], center["pre"])]
            assert all(abs(value) <= radius for value, radius in zip(difference, center["delta"]))
            remainder = [
                actual_value - center_value - gate * error
                for actual_value, center_value, gate, error in
                zip(actual["output"], center["output"], center["gates"], difference)
            ]
            assert all(Q(0) <= value <= ramp for value, ramp in zip(remainder, center["ramp"]))
            remainders.append(remainder)
        for readout, _ in readouts:
            suffixes, input_gradient = adjoints(layers, cell["trace"], readout)
            actual_error = dot(readout, [x - y for x, y in zip(output, cell["output"])])
            reconstructed = dot(input_gradient, residual) + sum(
                (dot(suffix, remainder) for suffix, remainder in zip(suffixes, remainders)), Q(0)
            )
            assert actual_error == reconstructed
            identities += 1
    return mean, identities


def encoded(value):
    if isinstance(value, Q):
        return {"exact": str(value), "decimal": float(value)}
    raise TypeError(type(value).__name__)


def experiment(name, width, stable=False):
    started = time.perf_counter()
    layers = network(width, stable)
    summary = prepare(layers, width, 4)
    serialized = json.dumps(summary, default=encoded, sort_keys=True).encode()
    # Readouts are supplied only after the network summary is fixed.
    readouts = [([Q(1), Q(-1, 2)], Q(1, 7)),
                ([Q(0), Q(1)], Q(0)), ([Q(-2), Q(1)], Q(-1, 5))]
    if width == 8:
        mean, identities = exhaustive_reference(layers, summary, readouts)
        reference = {"method": "all independent sign inputs", "full_inputs_enumerated": 2 ** width,
                     "pointwise_error_identities_checked": identities}
    elif stable:
        mean, patterns = affine_reference(layers, width)
        reference = {"method": "global stable-gate affine collapse", "gate_patterns": patterns,
                     "full_inputs_enumerated": 0}
    else:
        mean, evaluated = binomial_reference(layers, width, 4)
        reference = {"method": "identical discarded columns and exact binomial weights",
                     "grouped_states_evaluated": evaluated, "full_inputs_enumerated": 0,
                     "represented_input_count": 2 ** width}
    observations = []
    for readout, bias in readouts:
        estimate_value, bound, lower, upper, uniform = estimate(layers, summary, readout, bias)
        true_value = dot(readout, mean) + bias
        difference = true_value - estimate_value
        assert lower <= difference <= upper and abs(difference) <= bound <= uniform
        observations.append({"readout": readout, "bias": bias,
                             "surrogate_expectation": estimate_value, "true_expectation": true_value,
                             "difference": difference, "absolute_error_bound": bound,
                             "signed_error_interval": [lower, upper],
                             "uniform_remainder_bound": uniform})
    return {
        "name": name, "input_signs": width, "retained_signs": 4, "discarded_signs": width - 4,
        "distribution": "All input signs are mutually independent and uniform on {-1,+1}.",
        "hidden_widths": [len(matrix) for matrix, _ in layers],
        "surrogate_states": len(summary["cells"]), "summary_excludes_readout": True,
        "summary_bytes": len(serialized), "summary_sha256": hashlib.sha256(serialized).hexdigest(),
        "nonzero_discarded_first_layer_weights": sum(
            weight != 0 for row in layers[0][0] for weight in row[4:]),
        "positive_ramp_units_across_cells": sum(
            ramp > 0 for cell in summary["cells"] for layer in cell["trace"] for ramp in layer["ramp"]),
        "reference": reference, "readouts": observations,
        "wall_seconds_including_reference": time.perf_counter() - started,
    }


def correlated_counterexample():
    layers = [([[Q(2), Q(1)]], [Q(0)])]
    summary = prepare(layers, 2, 1)
    prediction, bound, _, _, _ = estimate(layers, summary, [Q(1)], Q(0))
    actual = sum((forward(layers, [a, a])[0][0] / 2 for a in (Q(-1), Q(1))), Q(0))
    assert bound == 0 and actual - prediction == Q(1, 2)
    return {
        "network": "ReLU(2*A + R)", "joint_distribution": "A=R, each sign equally likely",
        "marginal_discarded_mean": Q(0), "conditional_discarded_means": [Q(-1), Q(1)],
        "conditional_centering_satisfied": False,
        "surrogate_expectation": prediction, "true_expectation": actual,
        "incorrectly_applied_boundary_bound": bound, "actual_error": actual - prediction,
        "reason": "The fixed-gate input-linear term has nonzero conditional expectation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = {
        "scope": "Constructed exact-rational networks; no trained-model or speedup claim.",
        "execution": "Python Fraction checks, not kernel-produced proofs of these artifacts.",
        "bound": "mean over retained cells of sum_l,i abs(suffix_l,i)*max(0,delta_l,i-abs(pre_l,i))",
        "centering": "Independent discarded signs have zero mean conditional on retained signs.",
        "cases": [experiment("small_exhaustive", 8), experiment("large_stable", 64, True),
                  experiment("large_nonlinear", 64)],
        "correlated_counterexample": correlated_counterexample(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    text = json.dumps(result, indent=2, default=encoded) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
