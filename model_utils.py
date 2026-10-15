from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_PATH = Path(__file__).parent / "models" / "plant_disease_model.pth"
PRETRAINED_MODEL_ID = "Kathir56/plant-disease-tamilnadu"


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def create_model(class_count: int, pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
    return model


def format_class_name(label: str) -> tuple[str, str]:
    """Return a plant name and condition from a PlantVillage-style class label."""
    if "___" in label:
        plant, condition = label.split("___", maxsplit=1)
    else:
        plant, condition = "", label
    return (
        plant.replace("_", " ").replace(",", "").strip(),
        condition.replace("_", " ").strip(),
    )


def load_classifier(model_source: str):
    """Load a locally trained checkpoint or the default pretrained classifier."""
    source = Path(model_source)
    if source.is_file():
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        labels = checkpoint.get("labels")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(labels, list) or len(labels) < 2 or not state_dict:
            raise ValueError("The model file is missing its class labels or model weights.")

        model = create_model(len(labels))
        model.load_state_dict(state_dict)
        model.eval()
        transform = image_transform()
        return model, labels, lambda image: transform(image).unsqueeze(0)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_source)
    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.eval()
    labels = [
        model.config.id2label[index]
        for index in range(model.config.num_labels)
    ]
    return model, labels, lambda image: processor(
        images=image, return_tensors="pt"
    )["pixel_values"]
