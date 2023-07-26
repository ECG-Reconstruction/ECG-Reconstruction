"""Utility functions for processing tree-structured configuration."""

import itertools
from collections.abc import Mapping


def deep_merge(map_a: Mapping, map_b: Mapping) -> dict:
    """Deep-merges two mappings."""

    def merge_impl(map_a, map_b):
        if not isinstance(map_a, Mapping) or not isinstance(map_b, Mapping):
            return map_b
        keys = set(itertools.chain(map_a.keys(), map_b.keys()))
        merged_dict = {}
        for key in keys:
            in_a = key in map_a
            in_b = key in map_b
            if in_a and in_b:
                merged_value = merge_impl(map_a[key], map_b[key])
            elif in_a:
                merged_value = map_a[key]
            elif in_b:
                merged_value = map_b[key]
            else:
                assert False, "Unreachable"
            merged_dict[key] = merged_value
        return merged_dict

    return merge_impl(map_a, map_b)
