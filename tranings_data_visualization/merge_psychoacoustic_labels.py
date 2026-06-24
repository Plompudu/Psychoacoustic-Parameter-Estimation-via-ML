import csv
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

LABELS_DIR = os.path.join(
    _ROOT, "data", "standardized_audio_files", "training_set",
    "psycho_acoustic_parameter_labels"
)
OUTPUT_DIR = os.path.join(
    _ROOT, "data", "standardized_audio_files", "training_set", "visualization"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_psychoacoustic_labels.csv")

CSV_FIELDS = [
    "time_index",
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]
OUTPUT_FIELDS = ["source_file"] + CSV_FIELDS

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pattern = os.path.join(LABELS_DIR, "*.csv")
    files = sorted(glob.glob(pattern))
    total = len(files)
    print(f"Found {total} CSV files to merge")

    written = 0
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, fpath in enumerate(files, 1):
            fname = os.path.basename(fpath)
            with open(fpath, newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    out_row = {"source_file": fname}
                    for field in CSV_FIELDS:
                        val = row.get(field, "")
                        if val == "" or val is None:
                            val = ""
                        out_row[field] = val
                    writer.writerow(out_row)
                    written += 1

            if i % 5000 == 0:
                print(f"  Processed {i}/{total} files ({written} rows written)")

    print(f"\nDone! Merged {total} files into {OUTPUT_FILE}")
    print(f"Total rows: {written}")

if __name__ == "__main__":
    main()
