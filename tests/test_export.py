from prodml.export import verify_onnx_parity


def test_onnx_parity():
    parity_results = verify_onnx_parity(num_samples=10)
    assert "max_prob_difference" in parity_results
    assert parity_results["max_prob_difference"] < 0.05
    assert parity_results["mismatch_count"] == 0
