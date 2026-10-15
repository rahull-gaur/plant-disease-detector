# Plant Disease Detector

A Python Streamlit app for uploading a plant-leaf photo and classifying it with a pretrained 38-class MobileNetV2 model.

## Run locally

Use Python 3.12 or newer from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the model from [Hugging Face](https://huggingface.co/Kathir56/plant-disease-tamilnadu) and caches it. If `models/plant_disease_model.pth` exists, the app uses that custom checkpoint instead.

## Deploy for phone access

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **Create app**, choose `rahull-gaur/plant-disease-detector`, branch `main`, and entrypoint `app.py`.
3. Deploy, then open the generated `https://...streamlit.app` address on your phone.

The repository includes `runtime.txt`, CPU PyTorch dependencies, and Streamlit configuration. Do not upload `.venv`, datasets, model checkpoints, or secrets.

## Optional: train your own model

Arrange a labeled image dataset as one directory per class, such as `PlantVillage/Apple___Apple_scab/*.jpg` and `PlantVillage/Apple___healthy/*.jpg`. Then run:

```powershell
python train_model.py --data PlantVillage --epochs 5
```

The best validation checkpoint is saved at `models/plant_disease_model.pth`.

## Limitations

The pretrained classifier covers 38 PlantVillage categories. The publisher reports 97.31% test accuracy; performance on real field photos may be lower, and confidence is not a guarantee of correctness. Treat predictions as a guide and confirm suspected disease with an agricultural expert.
