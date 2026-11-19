"""Train a real image-input model after collecting labeled Morse hand images."""
import argparse
import json
from image_learning import train_image_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-by", choices=["session", "subject"], default="session")
    args = parser.parse_args()
    print(json.dumps(train_image_model(split_by=args.split_by), ensure_ascii=False, indent=2))
