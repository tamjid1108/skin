import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from models.acne.acne_model import MyNet
from recommender.recommend import Recommender

products_path = 'recommender/essential_skin_care.csv'
features = ['normal',
            'dry',
            'oily',
            'combination',
            'sensitive',
            'general care',
            'hydration',
            'dull skin',
            'dryness',
            'softening',
            'smoothening',
            'fine lines',
            'wrinkles',
            'acne',
            'blemishes',
            'pore care',
            'daily use',
            'dark circles']

# Load the models
acne_model = torch.load('saved_models/acne-severity/best_6.pt',
                        map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
acne_model.eval()

skintype_model = torch.load('saved_models/skintype/best_20.pt',
                            map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
skintype_model.eval()

recommender = Recommender(products_path, features)


def predict_acne_level(image):
    # Define image preprocessing
    preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # If GPU is available, move the input to GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        acne_model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = acne_model(input_batch)

    # Post-process the output
    # applying Softmax to results
    prob = nn.Softmax(dim=1)
    probs = prob(output)
    # print(probs)
    predicted_class = torch.argmax(probs, axis=1).tolist()[0]
    return predicted_class


def predict_skin_type(img):
    preprocess = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    img = img.convert("RGB")
    img = preprocess(np.array(img))
    img = img.view(1, 3, 224, 224)

    if torch.cuda.is_available():
        img = img.cuda()

    with torch.no_grad():
        out = skintype_model(img)
        return out.argmax(1).item()


def get_feature_vector(skin_type, acne_level, selected_features):
    fv = [0]*len(features)

    if skin_type == 'Normal':
        fv[0] = 1
        fv[6] = 1
    elif skin_type == 'Dry':
        fv[1] = 1
        fv[6] = 1
        fv[8] = 1
    else:
        fv[2] = 1
        fv[3] = 1

    if acne_level in ['Severe', 'Extreme']:
        fv[3] = 1
        fv[4] = 1
        fv[13] = 1
    elif acne_level == "Mild":
        fv[13] = 1

    for feature in selected_features:
        fv[features.index(feature)] = 1

    return fv


def get_essential_skincare(fv):
    return recommender.recommend(fv)


def generate_visual_meter(severity_level):
    meter_width = 90

    meter_height = 8
    colors = ['green', 'yellow', 'orange', 'red']
    progress = 10 + (severity_level)/3 * meter_width
    progress_color = colors[severity_level]

    # Define HTML/CSS for the visual meter
    meter_html = f"""
        <div style="width: {meter_width}%; height: {meter_height}px; border-radius: 10px; background-color: #f1f1f1; position: relative;">
            <div style="width: {progress}%; height: {meter_height}px; border-radius: 10px; background-color: {progress_color}; position: absolute;"></div>
        </div>
    """
    return meter_html


# Main Streamlit app
def main():
    st.title("Skin Care Essentials")
    st.subheader("Based on your personal details")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    # If an image is uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict the acne severity level
        acne_level = predict_acne_level(image)

        # Display the predicted severity level using a progress bar
        severity_levels = ['Clear', 'Mild', 'Severe', 'Extreme']
        severity_level = severity_levels[acne_level]
        st.subheader("We've noticed the following from your picture")
        st.subheader("Acne Severity Level")
        st.write(severity_level)

        # Display the severity level using a progress bar
        meter_html = generate_visual_meter(acne_level)
        st.markdown(meter_html, unsafe_allow_html=True)

        # Predict the skin type
        skin_types = ['Dry', 'Normal', 'Oily']
        skin_type = skin_types[predict_skin_type(image)]

        st.subheader("Skin Type")

        st.write(f"{skin_type}", unsafe_allow_html=True)

        # selected_features = st.multiselect("Select features:", features)

        st.subheader("Select other concerns/expectations (if any)")
        # Display checkboxes for feature selection
        selected_features = []
        with st.form("Feature Selection"):
            features_to_ask = ['general care', 'dull skin', 'softening', 'smoothening',
                               'fine lines', 'wrinkles', 'blemishes', 'pore care', 'daily use', 'dark circles']
            for feature in features_to_ask:
                selected = st.checkbox(feature)
                if selected:
                    selected_features.append(feature)
            submitted = st.form_submit_button("Submit")

        if submitted:
            # st.write(selected_features)
            fv = get_feature_vector(
                skin_type, severity_level, selected_features)

            data = get_essential_skincare(fv)

            for category, products in data.items():
                st.header(category.capitalize())
                num_cols = 2
                num_products = len(products)
                num_rows = (num_products + num_cols - 1) // num_cols

                cols = st.columns(num_cols)
                for i in range(num_products):
                    col_index = i % num_cols
                    row_index = i // num_cols
                    with cols[col_index]:
                        # st.subheader(products[i]['name'])
                        # st.write(f"Brand: {products[i]['brand']}")
                        # st.write(f"Price: {products[i]['price']}")
                        # st.write(f"Skin Type: {products[i]['skin type']}")
                        # st.write(f"Concerns: {', '.join(products[i]['concern'])}")
                        # st.write(f"[Buy now]({products[i]['url']})")
                        st.markdown(
                            f"""<div style='
                        background-color: #f1f1f1;
                        color: black;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    '>
                    <h3 style="color:black;">{products[i]['name'].title()}</h3>
                    <p>By {products[i]['brand'].title()}</p>
                    <img src={products[i]['image_url']} style="max-width:100%; border-radius:10px; margin-bottom:10px;" alt="Product Image">
                    <p>Skin Type: {products[i]['skin type']}</p>
                    <p>Concerns: {', '.join(products[i]['concern'])}</p>
                    <p style='font-size: 28px;'>{products[i]['price']}</p>
                    <a style='
                        display: inline-block;
                        background-color: red;
                        color: white;
                        padding: 7px 20px 10px 20px;
                        border-radius: 20px;
                        text-decoration: none;
                        margin-top: 5px;
                        text-align: center;
                    ' href="{products[i]['url']}">Buy now</a>
                    </div>
                    </div>
                    """,
                            unsafe_allow_html=True
                        )


if __name__ == '__main__':
    main()
