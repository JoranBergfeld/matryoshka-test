from src.data import build_transforms, select_quick_classes, resolve_imagenet_path


def test_train_and_val_transforms_differ():
    train_t = build_transforms(train=True, image_size=224, resize_size=256,
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_t = build_transforms(train=False, image_size=224, resize_size=256,
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_names = [type(t).__name__ for t in train_t.transforms]
    val_names = [type(t).__name__ for t in val_t.transforms]
    assert "RandomResizedCrop" in train_names
    assert "RandomHorizontalFlip" in train_names
    assert "CenterCrop" in val_names
    assert "RandomHorizontalFlip" not in val_names


def test_quick_class_selection_deterministic():
    a = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=42)
    b = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=42)
    c = select_quick_classes(all_classes=list(range(1000)), num_classes=100, seed=7)
    assert a == b
    assert a != c
    assert len(a) == 100


def test_resolve_imagenet_path_missing_env_raises(monkeypatch):
    import pytest
    monkeypatch.delenv("IMAGENET_PATH", raising=False)
    with pytest.raises(RuntimeError) as e:
        resolve_imagenet_path("IMAGENET_PATH")
    assert "IMAGENET_PATH" in str(e.value)
