from src.compiled_counter import verify_counter


def test_counter8_exhaustive_cpu() -> None:
    result = verify_counter(8, exhaustive=True, device="cpu")
    assert result["passed"]
    assert result["checked"] == 256
