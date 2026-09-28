import os

DATASET_PATH = r"C:\Users\archa\Downloads\archive (1)\indoorCVPR_09\Images"

print("=== MIT Indoor Scenes Dataset ===")

if os.path.exists(DATASET_PATH):
    print("Dataset found successfully! ✅")

    categories = os.listdir(DATASET_PATH)

    print("\nNumber of categories:", len(categories))

    print("\nFirst 10 categories:")
    for category in categories[:10]:
        print("-", category)

else:
    print("Dataset NOT found ❌")