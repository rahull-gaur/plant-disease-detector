import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model_utils import IMAGE_SIZE, MEAN, MODEL_PATH, STD, create_model


def split_indices(targets, validation_fraction: float, seed: int):
    randomizer = random.Random(seed)
    by_class = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)

    train_indices, validation_indices = [], []
    for indices in by_class.values():
        randomizer.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * validation_fraction)),
        ) if len(indices) > 1 else 0
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    if not validation_indices:
        raise ValueError(
            "Not enough images for a validation split. Add at least two images "
            "from one or more classes."
        )
    randomizer.shuffle(train_indices)
    randomizer.shuffle(validation_indices)
    return train_indices, validation_indices


def main():
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV3-Small plant disease classifier."
    )
    parser.add_argument("--data", type=Path, default=Path("PlantVillage"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random weights without downloading ImageNet weights.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("--epochs and --batch-size must be positive; --learning-rate must be > 0.")
    if not args.data.is_dir():
        parser.error(f"Dataset folder not found: {args.data}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    train_data = datasets.ImageFolder(args.data, transform=train_transform)
    validation_data = datasets.ImageFolder(args.data, transform=validation_transform)
    if len(train_data.classes) < 2:
        parser.error("The dataset must contain at least two class subfolders.")
    if len(train_data) < 2:
        parser.error("The dataset must contain at least two images.")

    train_indices, validation_indices = split_indices(
        train_data.targets, validation_fraction=0.2, seed=args.seed
    )
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(train_data.classes), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}; classes: {len(train_data.classes)}")
    for epoch in range(args.epochs):
        model.train()
        loss_total, correct, seen = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += images.size(0)

        model.eval()
        validation_correct, validation_seen = 0, 0
        with torch.inference_mode():
            for images, targets in validation_loader:
                logits = model(images.to(device))
                validation_correct += (logits.argmax(dim=1) == targets.to(device)).sum().item()
                validation_seen += images.size(0)

        validation_accuracy = validation_correct / validation_seen
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={loss_total / seen:.4f} "
            f"train_accuracy={correct / seen:.3f} "
            f"validation_accuracy={validation_accuracy:.3f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "labels": train_data.classes,
                    "image_size": IMAGE_SIZE,
                },
                args.output,
            )

    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved model to {args.output.resolve()}")


if __name__ == "__main__":
    main()
