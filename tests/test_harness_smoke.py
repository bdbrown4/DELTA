"""Smoke tests for the experiment harness (experiments/phase46c_link_prediction).

The 44 library tests never import experiments/, so harness regressions (bad
imports, the new sparse sampler, the load_lp_data cache) would pass CI green.
These CPU-only smoke tests close that gap for the Phase 71 changes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', 'fb15k-237', 'train.txt')
needs_data = pytest.mark.skipif(not os.path.exists(_DATA),
                                reason="FB15k-237 not downloaded")


@needs_data
def test_random_sample_is_sparser_than_degree():
    """sample_mode='random' must yield a much sparser subgraph than the
    default top-degree subgraph at the same entity budget."""
    from experiments.phase46c_link_prediction import (
        load_lp_data, build_train_graph_tensors)

    N = 1500
    dense = load_lp_data('fb15k-237', 'data', max_entities=N, sample_mode='degree')
    sparse = load_lp_data('fb15k-237', 'data', max_entities=N, sample_mode='random',
                          sample_seed=42)

    def mean_deg(d):
        ei, _ = build_train_graph_tensors(d['train'])
        return 2 * ei.shape[1] / d['num_entities']

    md_dense, md_sparse = mean_deg(dense), mean_deg(sparse)
    assert md_sparse < md_dense, (md_sparse, md_dense)
    # Sanity: sparse regime should be in the Phase 67/68 ballpark (single digits),
    # dense top-N is ~20-40.
    assert md_sparse < 15 < md_dense


@needs_data
def test_random_sample_deterministic_by_seed():
    from experiments.phase46c_link_prediction import load_lp_data
    a = load_lp_data('fb15k-237', 'data', max_entities=800, sample_mode='random',
                     sample_seed=7)
    b = load_lp_data('fb15k-237', 'data', max_entities=800, sample_mode='random',
                     sample_seed=7)
    assert a['num_entities'] == b['num_entities']
    assert a['train'].shape == b['train'].shape


@needs_data
def test_load_lp_cache_roundtrip(tmp_path):
    """use_cache round-trips identical entity/edge counts."""
    from experiments.phase46c_link_prediction import load_lp_data
    kw = dict(max_entities=600, sample_mode='random', sample_seed=3, use_cache=True)
    first = load_lp_data('fb15k-237', 'data', **kw)
    second = load_lp_data('fb15k-237', 'data', **kw)  # served from cache
    assert first['num_entities'] == second['num_entities']
    assert first['train'].shape == second['train'].shape
    assert first['test'].shape == second['test'].shape


if __name__ == '__main__':
    test_random_sample_is_sparser_than_degree()
    test_random_sample_deterministic_by_seed()
    print("harness smoke tests passed")
