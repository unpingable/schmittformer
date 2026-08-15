from pathlib import Path

from experiments.run_semantic_register_sweep import adaptive_eval_shape, completed, load_manifest, mark_completed, save_manifest


def test_adaptive_eval_shape_reduces_4096_batch() -> None:
    batch_64, batches_64 = adaptive_eval_shape(64, 512, 4)
    batch_4096, batches_4096 = adaptive_eval_shape(4096, 512, 4)
    assert batch_64 == 512
    assert batch_4096 < batch_64
    assert batch_4096 >= 4
    assert batches_4096 >= 1


def test_semantic_register_manifest_resume(tmp_path: Path) -> None:
    manifest = load_manifest(tmp_path)
    save_manifest(tmp_path, manifest)
    run_path = tmp_path / "runs" / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("{}")
    mark_completed(tmp_path, manifest, "run", run_path)
    reloaded = load_manifest(tmp_path)
    assert completed(reloaded, "run")
    assert reloaded["runs"]["run"] == str(run_path)
