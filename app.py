import streamlit as st
from PIL import Image
import io

# Import your local backend class
from Cloaker import AegisCloak

# Initialize backend once
engine = AegisCloak()

st.set_page_config(page_title="PANE", page_icon="🛡️")
st.title("🛡️ PANE: Local Cloaking System (CPU Mode)")

# Upload file
uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

# Settings
steps = st.slider("Optimization Steps", 5, 60, 30)
epsilon = st.slider("Epsilon (Perturbation Strength)", 0.01, 0.20, 0.03)

# When file is uploaded:
if uploaded_file is not None:

    # Show original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    # Convert image → bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Run attack
    if st.button("Run PANE"):
        with st.spinner("Running cloaking engine... (CPU may take time)"):

            # Direct call (NO HTTP)
            output_img = engine.vanish(img_bytes, steps, epsilon)

            # Display
            with col2:
                st.subheader("Cloaked Image")
                st.image(output_img, use_container_width=True)

                # Prepare download
                out_buf = io.BytesIO()
                output_img.save(out_buf, format="PNG")
                out_bytes = out_buf.getvalue()

                st.download_button(
                    "Download Cloaked Image",
                    out_bytes,
                    "cloaked.png",
                    "image/png"
                )

                st.success("Cloaking Successful!")
