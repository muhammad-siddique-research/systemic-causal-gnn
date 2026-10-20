"""Optional PyTorch tests for train/test separation by simulated graph."""
import pytest
torch = pytest.importorskip("torch")
from src.research.gnn_benchmark import generate_graphs, evaluate_gnn
from src.models.spatio_temporal_causal_gnn import SpatioTemporalCausalGNN

def test_graph_sequence_shapes():
    samples = generate_graphs(count=12, nodes=8, periods=3, seed=20)
    _, x, adj, y = samples[0]
    assert x.shape == (3, 8, 6)
    assert adj.shape == (3, 8, 8)
    assert y.shape == (8,)
    assert torch.all(torch.diagonal(adj, dim1=1, dim2=2) > 0)

def test_gnn_inference_shapes():
    _, x, adj, y = generate_graphs(count=12, nodes=8, seed=21)[0]
    model = SpatioTemporalCausalGNN(in_features=6, hidden_dim=12)
    model.eval()
    with torch.no_grad():
        pred, untrained_head, embed = model(x, adj)
    assert pred.shape == (8, 1)
    assert embed.shape == (8, 12)
    assert torch.isfinite(pred).all()

def test_independent_graph_heldout_training(tmp_path):
    result = evaluate_gnn(count=20, nodes=10, epochs=3, seed=42, output=tmp_path)
    assert result["train_networks"] == 15
    assert result["test_networks"] == 5
    assert 0 <= result["test_roc_auc"] <= 1
    assert (tmp_path / "gnn_heldout_metrics.json").exists()
    assert (tmp_path / "gnn_out_of_network.png").exists()
