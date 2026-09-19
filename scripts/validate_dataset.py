from pathlib import Path
from PIL import Image

DATASET_DIR = Path(r"C:\Users\DELL\Downloads\SmartMedWaste-Dataset")

SPLITS = ["train", "validation", "test"]

CLASSES = [
    "ANATOMICAL",
    "CONTAMINATED_RECYCLABLE",
    "SHARPS",
    "PHARMA_GLASS"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def validate_dataset():
    print("=" * 70)
    print("SMARTMEDWASTE DATASET VALIDATION")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset not found:")
        print(DATASET_DIR)
        return

    total_images = 0
    total_errors = 0

    for split in SPLITS:
        print(f"\n{'-' * 70}")
        print(f"{split.upper()} DATASET")
        print(f"{'-' * 70}")

        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            print(f"ERROR: Missing folder: {split_dir}")
            total_errors += 1
            continue

        split_total = 0

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"ERROR: Missing class folder: {class_name}")
                total_errors += 1
                continue

            images = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]

            valid_count = 0
            invalid_count = 0
            min_width = None
            min_height = None
            max_width = None
            max_height = None

            for image_path in images:
                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    with Image.open(image_path) as img:
                        width, height = img.size

                    valid_count += 1

                    min_width = width if min_width is None else min(min_width, width)
                    min_height = height if min_height is None else min(min_height, height)

                    max_width = width if max_width is None else max(max_width, width)
                    max_height = height if max_height is None else max(max_height, height)

                except Exception as e:
                    invalid_count += 1
                    total_errors += 1

                    print(
                        f"  INVALID: {image_path.name} -> {e}"
                    )

            split_total += valid_count

            print(
                f"{class_name:<28} "
                f"Images: {valid_count:<5} "
                f"Invalid: {invalid_count:<3} "
                f"Size: {min_width}x{min_height} - "
                f"{max_width}x{max_height}"
            )

        print(f"\nTotal valid images in {split}: {split_total}")

        total_images += split_total

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Total valid images : {total_images}")
    print(f"Total errors       : {total_errors}")

    if total_errors == 0:
        print("\nSTATUS: DATASET VALIDATION PASSED")
        print("The dataset is ready for the training preparation stage.")
    else:
        print("\nSTATUS: DATASET HAS ERRORS")
        print("Fix the reported problems before training.")

    print("=" * 70)


if __name__ == "__main__":
    validate_dataset()