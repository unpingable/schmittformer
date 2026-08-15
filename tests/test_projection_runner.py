from experiments.run_projection_sweep import projection_points, profile_defaults, scaled_eval_shape
from src.projection_channels import ProjectionRegime


def test_projection_points_include_erasure_and_full_export() -> None:
    points = projection_points([0.0, 0.1])
    regimes = {point["regime"] for point in points}

    assert ProjectionRegime.P0_COMPLETE_ERASURE.value in regimes
    assert ProjectionRegime.P3_FULL_TRUSTED_EXPORT.value in regimes
    assert ProjectionRegime.P1_NOISY_EXPORT.value in regimes


def test_profile_defaults_include_long_context_axis() -> None:
    defaults = profile_defaults("overnight")

    assert defaults["context_lengths"] == [64, 256, 1024]


def test_scaled_eval_shape_reduces_batch_for_long_contexts() -> None:
    batch_64, batches_64 = scaled_eval_shape(64, 64, 512, 8)
    batch_1024, batches_1024 = scaled_eval_shape(64, 1024, 512, 8)

    assert batch_64 == 512
    assert batches_64 == 8
    assert batch_1024 < batch_64
    assert batches_1024 < batches_64
    assert batch_1024 >= 16
    assert batches_1024 >= 1
