import subprocess
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit import runtime
import altair as alt

# Supported activity classes (can be updated to match the final trained model labels)
ACTIVITY_CLASSES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# 1. CONNECTOR INTERFACE 
def predict_activity(image: Image.Image) -> dict:
    """Standard connector function between the GUI and the HAR model."""
    # TODO: Replace this placeholder with real model inference
    return {
        "predicted_class": "WALKING",
        "confidence": 85.00,
        "probabilities": {
            "WALKING": 85.00,
            "WALKING_UPSTAIRS": 6.00,
            "WALKING_DOWNSTAIRS": 4.00,
            "SITTING": 2.50,
            "STANDING": 1.50,
            "LAYING": 1.00,
        },
    }



# 2. GUI INTERFACE
def render_ui():
    """Renders the Streamlit photo-upload Human Activity Recognition interface."""
    st.set_page_config(
        page_title="HAR - Photo Activity Recognition",
        page_icon="🏃",
        layout="wide",
    )

    st.markdown(
            "<h1 style='color: #2b7de9; margin-bottom: 0;'>🏃 Human Activity Recognition (HAR)</h1>",
            unsafe_allow_html=True,
        )
    st.caption("Upload a photo of a person performing a daily activity to classify the action.")

    with st.sidebar:
        st.header("System Info")
        st.markdown(
            "- **Input:** Single Photo\n"
            "- **Supported Formats:** `.jpg`, `.jpeg`, `.png`, `.webp`"
        )
        st.markdown("---")
        st.markdown("**Target Activity Classes:**")
        for idx, label in enumerate(ACTIVITY_CLASSES):
            st.markdown(f"- `{idx+1}`: **{label}**")


    # Upper Row: Photo Upload & Preview 
    st.subheader("1️⃣ Upload Activity Photo")
    uploaded_file = st.file_uploader(
        "Choose an image file:",
        type=["jpg", "jpeg", "png", "webp"],
    )

    selected_image = None
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        with st.container(border=True):
            st.image(selected_image, caption=uploaded_file.name, use_container_width=True)
            w, h = selected_image.size
            st.caption(f"Resolution: `{w} × {h} px` | Mode: `{selected_image.mode}`")
    else:
        st.info("Upload a photo to view the predicted activity.")

    st.divider()

    # Lower Row: Prediction Output
    st.subheader("2️⃣ Recognition Result")
    if selected_image is not None:
        # Call the connector function
        result = predict_activity(selected_image)

        with st.container(border=True):
            st.metric(
                label="Predicted Human Activity",
                value=result["predicted_class"],
                delta=f"Confidence: {result['confidence']}%",
            )

        with st.container(border=True):
                st.markdown("**Class Probability Distribution (%)**")
                df_prob = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Activity Class", "Confidence (%)"],
                )
                df_prob["Label"] = df_prob["Confidence (%)"].map(lambda x: f"{x:.2f}%")
                df_prob["IsTop"] = df_prob["Activity Class"] == result["predicted_class"]
    
                # Horizontal bar chart sorted by probability descending
                bars = (
                    alt.Chart(df_prob)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X(
                            "Confidence (%):Q",
                            scale=alt.Scale(domain=[0, 115]),
                            title="Confidence (%)",
                        ),
                        y=alt.Y(
                            "Activity Class:N",
                            sort="-x",
                            title=None,
                        ),
                        color=alt.condition(
                            alt.datum.IsTop,
                            alt.value("#2b7de9"),  # Highlight top predicted class
                            alt.value("#94a3b8"),  # Neutral slate for other classes
                        ),
                        tooltip=["Activity Class", "Label"],
                    )
                )
        
                # Percentage text labels placed right next to each bar
                text = bars.mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    fontWeight="bold",
                ).encode(
                    text="Label:N"
                )
        
                chart = (bars + text).properties(height=280)
                st.altair_chart(chart, use_container_width=True)
    else:
        with st.container(border=True):
            st.markdown("**Waiting for Photo Input...**")
            st.caption("The detected activity and confidence distribution will appear here once an image is uploaded.")
            
def main():
    """Entry point for launching the Streamlit interface."""
    if runtime.exists():
        render_ui()
    else:
        script_path = str(Path(__file__).resolve())
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=False)


if __name__ == "__main__":
    main()
