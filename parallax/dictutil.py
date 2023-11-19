from collections import defaultdict


def swap_axes(a_dict, ax1: int, ax2: int):
    """
    Examples:
        Homogeneous cases:
            >>> import pprint
            >>> a_dict = {'a': {'1': {'x': 'a1x', 'y': 'a1y'}, '2': {'x': 'a2x', 'y': 'a2y'}},
            ...           'b': {'1': {'x': 'b1x', 'y': 'b1y'}, '2': {'x': 'b2x', 'y': 'b2y'}}}
            >>> pprint.pprint(swap_axes(a_dict, 0, 1))
            {'1': {'a': {'x': 'a1x', 'y': 'a1y'}, 'b': {'x': 'b1x', 'y': 'b1y'}},
             '2': {'a': {'x': 'a2x', 'y': 'a2y'}, 'b': {'x': 'b2x', 'y': 'b2y'}}}
            >>> pprint.pprint(swap_axes(a_dict, 0, 2))
            {'x': {'1': {'a': 'a1x', 'b': 'b1x'}, '2': {'a': 'a2x', 'b': 'b2x'}},
             'y': {'1': {'a': 'a1y', 'b': 'b1y'}, '2': {'a': 'a2y', 'b': 'b2y'}}}
            >>> pprint.pprint(swap_axes(a_dict, 1, 2))
            {'a': {'x': {'1': 'a1x', '2': 'a2x'}, 'y': {'1': 'a1y', '2': 'a2y'}},
             'b': {'x': {'1': 'b1x', '2': 'b2x'}, 'y': {'1': 'b1y', '2': 'b2y'}}}
        Heterogeneous case:
            >>> a_dict['a']['1'].pop('x')
            'a1x'
            >>> pprint.pprint(swap_axes(a_dict, 1, 2))
            {'a': {'x': {'2': 'a2x'}, 'y': {'1': 'a1y', '2': 'a2y'}},
             'b': {'x': {'1': 'b1x', '2': 'b2x'}, 'y': {'1': 'b1y', '2': 'b2y'}}}
    """
    if ax1 == ax2:
        return a_dict
    assert ax1 >= 0, f"Axis should not be negative (ax1={ax1})"
    assert ax2 >= 0, f"Axis should not be negative (ax2={ax2})"
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    if ax1 > 0:
        return {
            k: swap_axes(v, ax1 - 1, ax2 - 1)
            for k, v in a_dict.items()
        }
    # ax1 == 0
    ret = nested_defaultdict(depth=ax2)
    def callback(v, *ks):
        set_deep(ret, (ks[-1], *ks[1:-1], ks[0]), v)
    for_depth(a_dict, [], depth=ax2 + 1, callback=callback)
    return dictify(ret, depth=ax2)


def nested_defaultdict(depth: int):
    if depth == 0:
        return dict()
    return defaultdict(lambda: nested_defaultdict(depth - 1))


def dictify(a_dict, depth: int):
    if depth == 1:
        return dict(a_dict)
    else:
        return {
            k: dictify(v, depth - 1)
            for k, v in a_dict.items()
        }


def get_deep(a_dict, ks):
    if len(ks) == 1:
        return a_dict[ks[0]]
    return get_deep(a_dict[ks[0]], ks[1:])


def set_deep(a_dict, ks, v):
    if len(ks) == 1:
        a_dict[ks[0]] = v
    else:
        set_deep(a_dict[ks[0]], ks[1:], v)


def for_depth(v, ks, depth: int, callback):
    if depth == 0:
        callback(v, *ks)
    else:
        for k, _v in v.items():
            for_depth(_v, (*ks, k), depth - 1, callback)
