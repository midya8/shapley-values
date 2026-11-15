"""Finite checks of the supplied-game rational decomposition, not data validity."""
import copy
import itertools
import json
import math
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

from shapleylab.dividends import decompose_game
from shapleylab.owen_controls import definitions
from shapleylab.owen_report import audit_owen


def rational(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for part in partitions(rest):
        yield [[first]] + [row[:] for row in part]
        for k in range(len(part)):
            yield [[first] + row if k == q else row[:] for q, row in enumerate(part)]


def named(part):
    return [{'name': 'group ' + str(k), 'features': row} for k, row in enumerate(part)]


def order_reference(values, part):
    result = [Fraction() for _ in range(len(values).bit_length() - 1)]
    count = 0
    for outer in itertools.permutations(range(len(part))):
        for blocks in itertools.product(*(itertools.permutations(part[k]) for k in outer)):
            mask = 0
            count += 1
            for block in blocks:
                for j in block:
                    after = mask | 1 << j
                    result[j] += Fraction(values[after]) - Fraction(values[mask])
                    mask = after
    return [value / count for value in result], count


class DividendTests(unittest.TestCase):
    def check_reconstruction(self, result, values):
        self.assertEqual(list(map(rational, result['reconstructed_values'])), list(map(Fraction, values)))
        self.assertEqual(list(map(rational, result['reconstruction_residuals'])), [0] * len(values))
        self.assertEqual(rational(result['ordinary_efficiency_residual']), 0)
        self.assertEqual(rational(result['owen_efficiency_residual']), 0)
        self.assertEqual(json.loads(json.dumps(result)), result)
        for item in result['interactions']:
            self.assertEqual(sum(map(rational, item['ordinary_contributions'])), rational(item['dividend']))
            self.assertEqual(sum(map(rational, item['owen_contributions'])), rational(item['dividend']))

    def test_all_partitions_signed_dyadic_tables_and_independent_order_reference(self):
        rng = random.Random(192839)
        cases = paths = 0
        for n in range(1, 7):
            for part in partitions(list(range(n))):
                for exponent in (0, 3):
                    values = [rng.randrange(-32, 33) / (1 << exponent) for _ in range(1 << n)]
                    result = decompose_game(values, [str(j) for j in range(n)], named(part))
                    self.check_reconstruction(result, values)
                    expected, count = order_reference(values, part)
                    self.assertEqual(list(map(rational, result['owen_attributions'])), expected)
                    self.assertEqual(list(map(rational, result['owen_group_sums'])), [sum((expected[j] for j in row), Fraction()) for row in part])
                    ordinary, _ = order_reference(values, [list(range(n))])
                    self.assertEqual(list(map(rational, result['ordinary_attributions'])), ordinary)
                    # Separate inclusion/exclusion, not the in-place bit transform.
                    for interaction in result['interactions']:
                        mask = interaction['mask']
                        dividend = sum(((-1) ** ((mask ^ sub).bit_count()) * Fraction(values[sub])
                                        for sub in range(1 << n) if sub & mask == sub), Fraction())
                        self.assertEqual(rational(interaction['dividend']), dividend)
                    cases += 1
                    paths += count
        self.assertEqual(cases, 556)
        self.assertEqual(paths, 50362)

    def test_all_original_controls_and_no_generator_or_permutation_calls(self):
        for control in definitions():
            report = audit_owen(control['request'])
            allocation = report['allocation']
            result = decompose_game(allocation['coalition_values'], allocation['feature_names'], control['request']['groups'])
            self.check_reconstruction(result, allocation['coalition_values'])
            for actual, expected in zip(result['owen_attributions'], allocation['attributions']):
                self.assertTrue(math.isclose(float(rational(actual)), expected, rel_tol=1e-12, abs_tol=1e-12))
        with patch('shapleylab.owen.allocate_owen', side_effect=AssertionError('coefficient engine called')), patch('itertools.permutations', side_effect=AssertionError('orders enumerated')):
            result = decompose_game([0] * 7 + [1], list('abc'), named([[0, 1], [2]]))
        self.assertEqual(list(map(rational, result['owen_attributions'])), [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)])
        self.assertEqual(list(map(rational, result['ordinary_attributions'])), [Fraction(1, 3)] * 3)
        self.assertEqual(len(result['interactions']), 7)  # Zero dividends stay visible.

    def test_float_identity_cancellation_subnormal_and_eight_feature_boundaries(self):
        for values in ([0.1, 0.2, 0.3, 0.4], [1e308, -1e308, -1e308, 1e308], [0., 5e-324, -5e-324, 0.], [2 ** 1023] * 4):
            result = decompose_game(values, list('ab'), named([[1, 0]]))
            self.check_reconstruction(result, values)
            self.assertEqual(rational(result['coalition_values'][0]), Fraction(values[0]))
        self.assertNotEqual(Fraction(0.1), Fraction(1, 10))
        values = [7 + sum(j + 1 for j in range(8) if mask >> j & 1) + 3 * (mask & 7 == 7) for mask in range(256)]
        for part in ([[j] for j in range(8)], [list(range(8))], [[0, 2, 4, 6], [1, 3, 5, 7]]):
            result = decompose_game(values, [str(j) for j in range(8)], named(part))
            expected, _ = order_reference(values, part)
            self.assertEqual(list(map(rational, result['owen_attributions'])), expected)
            self.check_reconstruction(result, values)
            self.assertEqual(len(result['interactions']), 255)

    def test_strict_inputs_no_alias_and_relabeling(self):
        values = [0, 1, 2, 5, 4, 9, -2, 11]
        names, groups = list('abc'), named([[0, 1], [2]])
        frozen = copy.deepcopy((values, names, groups))
        result = decompose_game(values, names, groups)
        self.assertEqual((values, names, groups), frozen)
        order = [2, 0, 1]
        permuted = [values[sum(1 << order[j] for j in range(3) if mask >> j & 1)] for mask in range(8)]
        changed = decompose_game(permuted, list('cab'), named([[0], [2, 1]]))
        self.assertEqual(changed['owen_attributions'], [result['owen_attributions'][i] for i in order])
        result['feature_names'][0] = 'changed'
        result['group_feature_indices'][0].append(7)
        self.assertEqual((values, names, groups), frozen)
        bad_values = [tuple(values), values[:-1], [True] + values[1:], [float('nan')] + values[1:], [float('inf')] + values[1:], [2 ** 1024] + values[1:], [Fraction(1, 2)] + values[1:]]
        for bad in bad_values:
            with self.assertRaises(ValueError):
                decompose_game(bad, names, groups)
        for bad in ([], ['a'] * 3, tuple(names), ['a', ' ', 'c'], list('abcdefghi')):
            with self.assertRaises(ValueError):
                decompose_game(values, bad, groups)
        for bad in ([], named([[0, 1]]), named([[0, 1], [1, 2]]), named([[True, 1], [2]]), named([[-1, 1], [0, 2]]), [{'name': 'x', 'features': [0, 1, 2], 'extra': 1}], [{'name': 'x', 'features': [0]}, {'name': 'x', 'features': [1, 2]}]):
            with self.assertRaises(ValueError):
                decompose_game(values, names, bad)


if __name__ == '__main__':
    unittest.main()
