# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

"""Property tests for ``VersionRange.to_specifier_set`` round-tripping.

The conversion is partial (not every range is specifier-expressible),
but when it succeeds it must round-trip exactly. ``None`` is allowed;
silent semantic drift is not.

Every round-trip test uses :func:`assert_ranges_equivalent` instead of
bare ``==``, so all public methods are checked (structural equality,
filtering, containment, ``is_empty``, ``is_prerelease_only``).
Hypothesis generates diverse specifier sets and version lists; the
behavioural oracle surfaces drift automatically.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from packaging.ranges import VersionRange

from .strategies import (
    SETTINGS,
    VERSION_POOL,
    assert_ranges_equivalent,
    specifier_sets,
)

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

pytestmark = pytest.mark.property

# List of 2–5 versions from the diverse VERSION_POOL.
# Two or more are needed because PEP 440 pre-release buffering only
# exposes a ``_prereleases`` divergence when a pre-release and a final
# release appear in the same batch.
_version_list = st.lists(st.sampled_from(VERSION_POOL), min_size=2, max_size=5)


@given(spec_set=specifier_sets(), versions=_version_list)
@SETTINGS
def test_specifier_derived_ranges_always_have_a_specifier_set(
    spec_set: SpecifierSet, versions: list[str],
) -> None:
    """Specifier-derived ranges always re-encode (incl. ``<0`` for empty)."""
    r = VersionRange.from_specifier_set(spec_set)
    converted = r.to_specifier_set()
    assert converted is not None, (
        f"specifier-derived range {r!r} should always re-encode "
        f"(input was {spec_set!r})"
    )
    assert_ranges_equivalent(
        VersionRange.from_specifier_set(converted), r, versions
    )


@given(spec_set=specifier_sets(), versions=_version_list)
@SETTINGS
def test_to_specifier_sets_round_trips_when_not_none(
    spec_set: SpecifierSet, versions: list[str],
) -> None:
    """If ``to_specifier_sets`` succeeds, the union of its elements equals ``r``."""
    r = VersionRange.from_specifier_set(spec_set)
    converted = r.to_specifier_sets()
    if converted is None:
        return
    assert converted, "to_specifier_sets must return a non-empty tuple"
    union = reduce(
        VersionRange.union,
        (VersionRange.from_specifier_set(s) for s in converted),
    )
    assert_ranges_equivalent(union, r, versions)


@given(a=specifier_sets(), b=specifier_sets(), versions=_version_list)
@SETTINGS
def test_intersection_round_trips_when_not_none(
    a: SpecifierSet, b: SpecifierSet, versions: list[str],
) -> None:
    """SpecifierSet is closed under intersection."""
    ra = VersionRange.from_specifier_set(a)
    rb = VersionRange.from_specifier_set(b)
    inter = ra & rb
    converted = inter.to_specifier_set()
    assert converted is not None
    assert_ranges_equivalent(
        VersionRange.from_specifier_set(converted), inter, versions
    )


@given(a=specifier_sets(), b=specifier_sets(), versions=_version_list)
@SETTINGS
def test_to_specifier_sets_handles_union_when_intervals_are_specifier_shaped(
    a: SpecifierSet, b: SpecifierSet, versions: list[str],
) -> None:
    """Per-interval encoding succeeds for unions of specifier-derived ranges."""
    ra = VersionRange.from_specifier_set(a)
    rb = VersionRange.from_specifier_set(b)
    u = ra | rb
    converted = u.to_specifier_sets()
    assert converted is not None
    union = reduce(
        VersionRange.union,
        (VersionRange.from_specifier_set(s) for s in converted),
    )
    assert_ranges_equivalent(union, u, versions)


@given(spec_set=specifier_sets())
@SETTINGS
def test_to_specifier_set_implies_to_specifier_sets(
    spec_set: SpecifierSet,
) -> None:
    """``to_specifier_set is not None`` ⇒ ``to_specifier_sets is not None``."""
    r = VersionRange.from_specifier_set(spec_set)
    if r.to_specifier_set() is not None:
        assert r.to_specifier_sets() is not None


@given(spec_set=specifier_sets(), versions=_version_list)
@SETTINGS
def test_complement_round_trips_or_returns_none(
    spec_set: SpecifierSet, versions: list[str],
) -> None:
    """The complement of a specifier-derived range is often not
    specifier-expressible (e.g. ``~(>=1,<2)`` is two disjoint intervals).
    Exercises the partial-conversion contract: ``to_specifier_set`` either
    returns ``None`` or round-trips exactly, never drifting.

    Uses :func:`assert_ranges_equivalent` (not bare ``==``) so that
    pre-release semantic drift is automatically detected.
    """
    r = VersionRange.from_specifier_set(spec_set).complement()
    converted = r.to_specifier_set()
    if converted is not None:
        assert_ranges_equivalent(
            VersionRange.from_specifier_set(converted), r, versions
        )
