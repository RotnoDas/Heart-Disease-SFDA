from pathlib import Path

import json


ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = (
    ROOT
    / "configs"
    / "final_experiment.json"
)


def load_config():

    assert CONFIG_FILE.exists()

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def test_experiment_is_frozen():

    config = load_config()

    assert (
        config["study"]["status"]
        == "frozen"
    )


def test_final_evaluation_settings():

    config = load_config()

    evaluation = (
        config["evaluation"]
    )

    assert (
        evaluation["source_seeds"]
        == 10
    )

    assert (
        evaluation["target_crossfit_folds"]
        == 5
    )

    assert (
        evaluation["bootstrap_iterations"]
        == 5000
    )

    assert (
        evaluation["classification_threshold"]
        == 0.5
    )


def test_source_epoch_plan():

    config = load_config()

    epochs = (
        config[
            "single_head_source_epochs"
        ]
    )

    assert epochs == {
        "cleveland": 34,
        "hungary": 32,
        "switzerland": 44,
        "va_long_beach": 33,
    }