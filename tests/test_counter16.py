from src.compiled_counter import verify_counter


def test_counter16_exhaustive_cpu() -> None:
    result = verify_counter(16, exhaustive=True, device="cpu")
    assert result["passed"]
    assert result["checked"] == 65536
