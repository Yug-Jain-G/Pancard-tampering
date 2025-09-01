import streamlit as st
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from PIL import Image
import imutils
import requests

st.set_page_config(page_title="PAN Card Tampering Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ PAN Card Tampering Detection")

st.write("Upload a PAN card image and the system will compare it against the original reference template.")

# --- Load reference (original) PAN template ---
@st.cache_resource
def load_reference():
    url = "https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg"
    ref_img = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((250,160))
    return ref_img

reference = load_reference()

# Show reference on sidebar
st.sidebar.subheader("Reference PAN Template")
st.sidebar.image(reference, caption="Original PAN Template", use_container_width=True)

# --- Upload suspected PAN card ---
uploaded_file = st.file_uploader("Upload Suspected PAN Card", type=["jpg", "jpeg", "png"])

if uploaded_file:
    suspected = Image.open(uploaded_file).convert("RGB").resize((250,160))

    st.subheader("Uploaded Image")
    st.image(suspected, caption="Suspected PAN Card", use_container_width=True)

    # Convert to OpenCV format
    reference_cv = cv2.cvtColor(np.array(reference), cv2.COLOR_RGB2BGR)
    suspected_cv = cv2.cvtColor(np.array(suspected), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference_cv, cv2.COLOR_BGR2GRAY)
    susp_gray = cv2.cvtColor(suspected_cv, cv2.COLOR_BGR2GRAY)

    # Structural Similarity
    (score, diff) = ssim(ref_gray, susp_gray, full=True)
    diff = (diff * 255).astype("uint8")
    st.info(f"Structural Similarity Index (SSIM): {score:.4f}")

    # Threshold
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(reference_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(suspected_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show results
    st.subheader("Detection Results")
    st.image([cv2.cvtColor(reference_cv, cv2.COLOR_BGR2RGB),
              cv2.cvtColor(suspected_cv, cv2.COLOR_BGR2RGB)],
             caption=["Reference with Highlights", "Suspected with Highlights"],
             use_container_width=True)

    st.subheader("Difference Map")
    st.image(diff, caption="SSIM Difference", use_container_width=True)

    st.subheader("Thresholded Map")
    st.image(thresh, caption="Binary Mask of Changes", use_container_width=True)
