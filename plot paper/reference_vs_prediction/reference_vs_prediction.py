import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))

from DL_model.reference_vs_prediction import plot_reference_vs_prediction_test  # noqa: E402

REFERENCE_PATH = _REPO / "data" / "reference_values" / "references.csv"
TEST_DIR = _REPO / "data" / "test_set"
CHECKPOINT_DIR = _REPO / "DL_model" / "epochs"
OUTPUT_DIR = _THIS

AUDIO_WORKERS = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=547,
                        help="checkpoint epoch to load (default: newest)")
    args = parser.parse_args()

    gathered = plot_reference_vs_prediction_test(
        TEST_DIR, REFERENCE_PATH, CHECKPOINT_DIR, OUTPUT_DIR,
        audio_workers=AUDIO_WORKERS, epoch=args.epoch,
    )
    if gathered is None:
        print("No checkpoint found in", CHECKPOINT_DIR)
        print("Run DL_model/train_model.py training first to produce a checkpoint.")
        return
    print("Point counts per parameter:", gathered)


if __name__ == "__main__":
    main()