import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import json

@st.cache_resource
def load_model():
    repo_id = "SharmarkeO/efficientnet-b0-sun10"

    # download config
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    id_to_label = {int(k): v for k, v in config["id_to_label"].items()}

    # build model
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, config["num_classes"])

    # download weights
    weights_path = hf_hub_download(repo_id, "pytorch_model.bin")
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

st.title("SUN10 Scene Classifier (EfficientNet‑B0)")

st.write(
    "This classifier was fine-tuned on **10 SUN scene categories only** "
    "and can only predict among those classes."
)

st.write(
    "Supported classes: beach, bedroom, kitchen, living room, office, "
    "mountain, highway, street, church indoor, forest broadleaf."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    model, id_to_label = load_model()

    # preprocess
    img_tensor = transform(image).unsqueeze(0)  # shape (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

    pred_label = id_to_label[int(pred)]
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {conf.item():.3f}")
    
    # --- Show all class confidences ---
    st.subheader("All Class Probabilities")
    
    # Build a sorted dataframe of all classes + their confidence
    all_probs = probs.numpy()  # shape: (num_classes,)
    prob_df = pd.DataFrame({
        "Class": [id_to_label[i] for i in range(len(all_probs))],
        "Confidence": all_probs
    }).sort_values("Confidence", ascending=False).reset_index(drop=True)

    st.bar_chart(prob_df.set_index("Class")["Confidence"])
