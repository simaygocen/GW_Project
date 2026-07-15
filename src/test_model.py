import numpy as np

from src.model import split_dataset_three_way


def test_three_way_split_is_complete_and_disjoint():
    n_samples = 100
    sample_ids = np.arange(n_samples)
    dataset = {
        "strain": sample_ids[:, None, None],
        "parameters": np.column_stack([sample_ids] * 6),
    }

    train_data, validation_data, test_data = split_dataset_three_way(
        dataset,
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=2026,
    )

    train_ids = set(train_data["strain"][:, 0, 0])
    validation_ids = set(validation_data["strain"][:, 0, 0])
    test_ids = set(test_data["strain"][:, 0, 0])

    assert len(train_ids) == 80
    assert len(validation_ids) == 10
    assert len(test_ids) == 10
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(sample_ids)
