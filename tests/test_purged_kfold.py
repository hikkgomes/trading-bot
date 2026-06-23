from src.walk_forward import generate_purged_kfold_windows


def test_generate_purged_kfold_windows_purges_around_test_fold():
    windows = generate_purged_kfold_windows(n_rows=1000, k=5, horizon=10, embargo=5)
    assert len(windows) == 5
    train_index, test_index = windows[2]
    assert train_index.size > 600
    assert test_index.size == 200
    assert not set(train_index).intersection(set(test_index))
    purge_start = max(0, int(test_index[0]) - 15)
    purge_end = min(1000, int(test_index[-1]) + 1 + 15)
    assert not set(train_index).intersection(set(range(purge_start, purge_end)))
