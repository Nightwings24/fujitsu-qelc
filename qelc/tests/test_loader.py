"""Loader semantics: per-token filtering, decimals, aggregation, SCC graph."""

import textwrap

import pytest

from qelc.data.loader import build_debt_graph, list_tokens, load_token_transfers

FIXTURE = textwrap.dedent(
    """\
    token_address,from_address,to_address,value
    0xAAA,0x1,0x2,1000000
    0xAAA,0x2,0x3,3000000
    0xAAA,0x3,0x1,2000000
    0xAAA,0x1,0x2,500000
    0xBBB,0x1,0x2,999999999999999999999
    0xBBB,0x9,0x1,123456789012345678901
    0xAAA,0x4,0x4,7000000
    0xAAA,0x5,0x6,4000000
    """
)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "transfers.csv"
    p.write_text(FIXTURE)
    return str(p)


def test_mixed_tokens_require_explicit_selection(csv_path):
    with pytest.raises(ValueError, match="multiple tokens"):
        load_token_transfers(csv_path)


def test_token_filter_decimals_and_aggregation(csv_path):
    edges = load_token_transfers(csv_path, token_address="0xaaa", decimals=6, min_amount=0.0)
    # only token AAA rows; 0x1->0x2 aggregated (1.0 + 0.5); self-loop dropped
    e = {(r.src, r.dst): r.weight for r in edges.itertuples()}
    assert e[("0x1", "0x2")] == pytest.approx(1.5)
    assert e[("0x2", "0x3")] == pytest.approx(3.0)
    assert e[("0x3", "0x1")] == pytest.approx(2.0)
    assert ("0x4", "0x4") not in e
    assert not any(src == "0x9" for src, _ in e)  # BBB rows excluded


def test_list_tokens(csv_path):
    tokens = list_tokens(csv_path)
    assert list(tokens["token"]) == ["0xaaa", "0xbbb"]
    assert int(tokens.loc[tokens["token"] == "0xaaa", "tx_count"].iloc[0]) == 6


def test_build_debt_graph_scc_restriction(csv_path):
    edges = load_token_transfers(csv_path, token_address="0xaaa", decimals=6, min_amount=0.0)
    G, node_map = build_debt_graph(edges)
    # the 1->2->3->1 cycle survives; the acyclic 5->6 edge does not
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3
    assert "0x5" not in node_map
