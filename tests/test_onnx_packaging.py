"""Hub blob links must not escape ONNX Runtime's external-data boundary."""

from pathlib import Path

from hyper_models.models import ONNXModel


def test_hub_symlinks_are_materialized_together(tmp_path, monkeypatch):
    import onnxruntime

    graph_blob = tmp_path / "blobs" / "graph" / "hash"
    weights_blob = tmp_path / "blobs" / "weights" / "hash"
    for path, content in ((graph_blob, b"graph"), (weights_blob, b"weights")):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.onnx").symlink_to(graph_blob)
    (snapshot / "model.onnx.data").symlink_to(weights_blob)
    seen = []

    def session(path, **kwargs):
        graph = Path(path)
        weights = graph.with_name("model.onnx.data")
        assert graph.resolve().parent == weights.resolve().parent
        assert graph.read_bytes() == b"graph"
        assert weights.read_bytes() == b"weights"
        assert kwargs == {"providers": ["CPUExecutionProvider"]}
        seen.append(graph)
        return object()

    monkeypatch.setattr(onnxruntime, "InferenceSession", session)
    model = ONNXModel(snapshot / "model.onnx", "hyperboloid", 513)
    model._ensure_session()
    model._ensure_session()
    assert len(seen) == 1
    assert seen[0].is_file()  # Retained for the session's lifetime.
    assert (snapshot / "model.onnx").is_symlink()  # Shared Hub cache stays intact.
