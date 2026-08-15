from pathlib import Path

from experiments.run_projection_sweep import completed, load_manifest, mark_completed, save_manifest


def test_manifest_records_completed_run(tmp_path: Path) -> None:
    manifest = load_manifest(tmp_path)
    save_manifest(tmp_path, manifest)
    run_path = tmp_path / "runs" / "seed.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text("{}")

    mark_completed(tmp_path, manifest, "seed-1", run_path)
    reloaded = load_manifest(tmp_path)

    assert completed(reloaded, "seed-1")
    assert reloaded["runs"]["seed-1"] == str(run_path)
