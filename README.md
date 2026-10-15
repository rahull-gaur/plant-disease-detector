# Plant Disease Detector

A Python Streamlit app for uploading leaf photos and classifying them with a pretrained 38-class plant disease model.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The first prediction downloads the pretrained model from Hugging Face. For phone access, deploy this repository using [Streamlit Community Cloud](https://share.streamlit.io/) and select `app.py`.

See the app README for training a custom classifier and model limitations.
