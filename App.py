import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import json
import pandas as pd

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import json
import pandas as pd

# ── Page config ──
st.set_page_config(
    page_title="SUN10 Scene Classifier",
    page_icon="🏞️",
    layout="wide"
)

REPO_ID = "SharmarkeO/sun10-scene-classifier"

MODELS = {
    "ResNet-18":       {"folder": "resnet18",        "build": "resnet18"},
    "EfficientNet-B0": {"folder": "efficientnet_b0", "build": "efficientnet_b0"},
    "MobileNetV3":     {"folder": "mobilenet_v3",    "build": "mobilenet_v3_large"},
}

CLASSES =  [
    "slum",
    "botanical garden",
    "arch",
    "volleyball court outdoor",
    "drugstore",
    "dentists office",
    "cottage garden",
    "cafeteria",
    "volcano",
    "boardwalk",
    "temple east asia",
    "playground",
    "bazaar indoor",
    "restaurant patio",
    "lake natural",
    "art school",
    "art gallery",
    "beauty salon",
    "corn field",
    "creek",
]


def build_model(build_key, num_classes):
    """Instantiate the correct architecture with a fresh classification head."""
    if build_key == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif build_key == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes)

    elif build_key == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, num_classes)

    return model


@st.cache_resource
def load_model(model_name):
    """Download weights + config from HuggingFace and return (model, id_to_label)."""
    cfg = MODELS[model_name]

    config_path = hf_hub_download(REPO_ID, f"{cfg['folder']}/config.json")
    with open(config_path) as f:
        config = json.load(f)

    id_to_label = {int(k): v for k, v in config["id_to_label"].items()}

    model = build_model(cfg["build"], config["num_classes"])

    weights_path = hf_hub_download(REPO_ID, f"{cfg['folder']}/pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, id_to_label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def run_inference(model, id_to_label, image):
    """Run a single image through the model. Returns (label, confidence, all_probs df)."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    pred_label = id_to_label[int(pred)]
    all_probs = probs.numpy()

    prob_df = pd.DataFrame({
        "Class":      [id_to_label[i] for i in range(len(all_probs))],
        "Confidence": all_probs.round(4)
    }).sort_values("Confidence", ascending=False).reset_index(drop=True)

    return pred_label, float(conf), prob_df


# UI
st.title("🏞️ SUN10 Scene Classifier")
st.caption(
    "Fine-tuned on 20 SUN397 scene categories: "
    + ", ".join(CLASSES)
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_results = st.columns([1, 2], gap="large")

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_results:
        st.subheader("Model Predictions")

        # Run all three models and display results side by side
        model_cols = st.columns(len(MODELS))

        for col, model_name in zip(model_cols, MODELS):
            with col:
                with st.spinner(f"Running {model_name}..."):
                    model, id_to_label = load_model(model_name)
                    pred_label, conf, prob_df = run_inference(model, id_to_label, image)

                st.metric(
                    label=model_name,
                    value=pred_label.title(),
                    delta=f"{conf:.1%} confidence"
                )

        st.divider()

        # Full confidence breakdown per model in expanders
        st.subheader("All Class Probabilities")
        for model_name in MODELS:
            with st.expander(f"{model_name} — full breakdown", expanded=True):
                model, id_to_label = load_model(model_name)   # cached — instant
                _, _, prob_df = run_inference(model, id_to_label, image)
                st.bar_chart(prob_df.set_index("Class")["Confidence"])
