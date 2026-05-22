from scripts.init_data import detect_credentials, dataset_present


def test_dataset_present_true_when_train_and_val_exist(tmp_path):
    (tmp_path / "train").mkdir(); (tmp_path / "val").mkdir()
    assert dataset_present(str(tmp_path)) is True


def test_dataset_present_false_when_missing(tmp_path):
    assert dataset_present(str(tmp_path)) is False


def test_detect_credentials_none(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("scripts.init_data._kaggle_json_exists", lambda: False)
    monkeypatch.setattr("scripts.init_data._hf_token_exists", lambda: False)
    assert detect_credentials() == []


def test_detect_credentials_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u"); monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setattr("scripts.init_data._hf_token_exists", lambda: False)
    assert "kaggle" in detect_credentials()
