from src.config import load_config, resolve_dim_weights


def test_load_default_returns_nested_dict():
    cfg = load_config("configs/default.yaml")
    assert cfg["run"]["seed"] == 42
    assert cfg["model"]["mrl_nesting_dims"][0] == 8


def test_cli_overrides_take_precedence():
    cfg = load_config("configs/default.yaml", overrides={"schedule.epochs": 3,
                                                          "run.seed": 7})
    assert cfg["schedule"]["epochs"] == 3
    assert cfg["run"]["seed"] == 7


def test_uniform_weighting_gives_equal_weights():
    dims = [8, 16, 32]
    assert resolve_dim_weights(dims, weighting="uniform", explicit=None) == [1.0, 1.0, 1.0]


def test_increasing_weighting_proportional_to_dim():
    dims = [8, 16, 32]
    w = resolve_dim_weights(dims, weighting="increasing", explicit=None)
    assert w == [1.0, 2.0, 4.0]


def test_explicit_weights_override_scheme():
    dims = [8, 16, 32]
    assert resolve_dim_weights(dims, weighting="increasing", explicit=[1.0, 1.0, 5.0]) == [1.0, 1.0, 5.0]


def test_explicit_weights_wrong_length_raises():
    import pytest
    with pytest.raises(ValueError):
        resolve_dim_weights([8, 16], weighting="uniform", explicit=[1.0])
