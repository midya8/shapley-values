"""Exact-rational interaction decomposition of a supplied finite game.

This API does not evaluate a model or validate the origin of coalition values.
Floats mean their exact stored binary values, not unknown pre-rounding reals.
"""
import math
from fractions import Fraction


def decompose_game(values, feature_names, groups):
    """Return every dividend and its ordinary/Owen feature contributions.

    At most eight features. Integer inputs are limited to 1024 bits; floats
    must be finite. Rational outputs use decimal-string numerator/denominator pairs.
    No allocation engine, factorial weights or permutation replay is called.
    """
    if type(feature_names) is not list or not 1 <= len(feature_names) <= 8:
        raise ValueError('require 1 through 8 feature names')
    if any(type(x) is not str or not x.strip() for x in feature_names) or len(set(feature_names)) != len(feature_names):
        raise ValueError('distinct nonempty feature names required')
    n = len(feature_names)
    if type(groups) is not list or not 1 <= len(groups) <= n:
        raise ValueError('nonempty partition required')
    names, members = [], []
    for group in groups:
        if type(group) is not dict or set(group) != {'name', 'features'}:
            raise ValueError('group requires name and features')
        name, row = group['name'], group['features']
        if type(name) is not str or not name.strip() or name in names:
            raise ValueError('distinct nonempty group names required')
        if type(row) is not list or not row or any(type(i) is not int or not 0 <= i < n for i in row):
            raise ValueError('nonempty valid integer group members required')
        names.append(name)
        members.append(row[:])
    if sorted(i for row in members for i in row) != list(range(n)):
        raise ValueError('groups must partition every feature exactly once')
    if type(values) is not list or len(values) != 1 << n:
        raise ValueError('complete coalition table required')
    for value in values:
        if type(value) is int:
            if value.bit_length() > 1024:
                raise ValueError('integer inputs are limited to 1024 bits')
        elif type(value) is not float or not math.isfinite(value):
            raise ValueError('finite integer or float game values required')
    exact = [Fraction(value) for value in values]
    dividends = exact[:]
    # Invert v(S) = sum_{T subset S} dividend(T), one bit at a time.
    for j in range(n):
        for mask in range(1 << n):
            if mask >> j & 1:
                dividends[mask] -= dividends[mask ^ (1 << j)]
    owner = {j: k for k, row in enumerate(members) for j in row}
    ordinary = [Fraction() for _ in range(n)]
    owen = ordinary[:]
    interactions = []
    def pair(value):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    for mask in range(1, 1 << n):
        active = [j for j in range(n) if mask >> j & 1]
        touched = sorted({owner[j] for j in active})
        counts = {k: sum(owner[j] == k for j in active) for k in touched}
        individual = [dividends[mask] / len(active) if j in active else Fraction() for j in range(n)]
        blocked = [dividends[mask] / (len(touched) * counts[owner[j]]) if j in active else Fraction() for j in range(n)]
        for j in range(n):
            ordinary[j] += individual[j]
            owen[j] += blocked[j]
        interactions.append({'mask': mask, 'features': active, 'touched_groups': touched,
                             'dividend': pair(dividends[mask]),
                             'ordinary_contributions': list(map(pair, individual)),
                             'owen_contributions': list(map(pair, blocked))})
    reconstructed = dividends[:]
    for j in range(n):
        for mask in range(1 << n):
            if mask >> j & 1:
                reconstructed[mask] += reconstructed[mask ^ (1 << j)]
    return {'method': 'exact_rational_dividend_decomposition',
            'feature_names': feature_names[:], 'group_names': names,
            'group_feature_indices': members,
            'coalition_values': list(map(pair, exact)),
            'baseline': pair(dividends[0]), 'prediction': pair(exact[-1]),
            'interactions': interactions,
            'ordinary_attributions': list(map(pair, ordinary)),
            'owen_attributions': list(map(pair, owen)),
            'owen_group_sums': [pair(sum((owen[j] for j in row), Fraction())) for row in members],
            'reconstructed_values': list(map(pair, reconstructed)),
            'reconstruction_residuals': [pair(a - b) for a, b in zip(reconstructed, exact)],
            'ordinary_efficiency_residual': pair(sum(ordinary, dividends[0]) - exact[-1]),
            'owen_efficiency_residual': pair(sum(owen, dividends[0]) - exact[-1]),
            'scope': 'Exact rational arithmetic on supplied integers or stored binary float values. '
                     'No model calls, source authentication, pre-rounding recovery, conditional or causal explanation. '
                     'Two-level partition only. Zero dividends do not certify valid background replacement.'}
