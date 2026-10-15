import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from model_utils import (
    MODEL_PATH,
    PRETRAINED_MODEL_ID,
    format_class_name,
    load_classifier as load_model,
)

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")


@st.cache_resource
def load_classifier(model_source: str):
    return load_model(model_source)


st.title("🌿 Plant Disease Detector")
st.write(
    "Upload a clear photo of a plant leaf to see the most likely condition predicted "
    "by an image-classification model."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=("jpg", "jpeg", "png", "webp", "bmp"),
    help="For best results, use a well-lit photo with one leaf in focus.",
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Could not read that image. Please upload a valid image file. ({exc})")
    else:
        st.image(image, caption="Uploaded plant image", use_container_width=True)

        try:
            model_source = str(MODEL_PATH) if MODEL_PATH.is_file() else PRETRAINED_MODEL_ID
            model, labels, preprocess = load_classifier(model_source)
            tensor = preprocess(image)
            with torch.inference_mode():
                output = model(tensor)
                logits = output.logits if hasattr(output, "logits") else output
                probabilities = torch.softmax(logits[0], dim=0)

            top_count = min(3, len(labels))
            scores, indices = torch.topk(probabilities, top_count)
            predicted_label = labels[indices[0].item()]
            plant, condition = format_class_name(predicted_label)

            st.subheader("Prediction")
            if plant:
                st.write(f"**Plant:** {plant}")
            st.write(f"**Most likely condition:** {condition}")
            st.metric("Model confidence", f"{scores[0].item() * 100:.1f}%")

            st.subheader("Other likely classes")
            for score, index in zip(scores, indices):
                label_plant, label_condition = format_class_name(labels[index.item()])
                display_name = (
                    f"{label_plant} — {label_condition}" if label_plant else label_condition
                )
                st.write(f"{display_name}: {score.item() * 100:.1f}%")

            if "healthy" in condition.casefold():
                st.info(
                    "The model's top prediction is healthy. This is an automated estimate, "
                    "not a substitute for expert inspection."
                )
            else:
                st.info(
                    "The model detected a possible disease. Consider isolating the plant "
                    "and confirming the result with a local agricultural expert."
                )

            st.caption(
                "The default classifier is pretrained on PlantVillage and covers 38 classes. "
                "Field photos can differ from its training data; confidence is not a guarantee."
            )
        except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
            st.error(f"Could not load or run the trained model: {exc}")

with st.sidebar:
    st.header("Getting started")
    st.write("Upload a leaf photo to use the pretrained 38-class model.")
    st.write("On its first run, the model is downloaded and cached automatically.")
    st.write("See `README.md` for instructions to train an optional custom model.")
