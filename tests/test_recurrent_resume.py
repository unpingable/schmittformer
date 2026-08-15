from experiments.run_recurrent_governance import load_or_compute_json


def test_load_or_compute_json_skips_completed_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"value": 1}')
    calls = []

    out = load_or_compute_json(path, False, lambda: calls.append("called") or {"value": 2})

    assert out == {"value": 1}
    assert calls == []


def test_load_or_compute_json_recomputes_with_force(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"value": 1}')

    out = load_or_compute_json(path, True, lambda: {"value": 2})

    assert out == {"value": 2}
    assert path.read_text().strip().startswith("{")
