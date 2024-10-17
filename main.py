import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_lottie import st_lottie
import time
import requests
import json

# Tensorflow model prediction
def set_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

    # Making UI for webapplication

st.sidebar.title("Dashboard")    
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Data and Matrix"])


# Function to load Lottie animation from a local JSON file
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animation from the local file
lottie_animation = load_lottie_file("Animation - 1725805951754.json")  # Replace with your local Lottie JSON file path

# Main logic
if(app_mode=='Home'):
    set_background("Black")
    st.markdown(
        """
        <h2 style='margin-left: 70px;'>Livestock And Plant Disease Predictor</h2>
        """,
        unsafe_allow_html=True
    ) 
    image_path="Collage_disease.jpg"
    st.image(image_path,use_column_width=True)

    st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
        font-size: 18px;
        line-height: 1.6;
        color: #f1f1f1;
    }
    .highlight {
        color: #39c5bb;
        font-weight: bold;
    }
    .important-text {
        font-size: 20px;
        color: #e84393;
        font-weight: bold;
    }
    </style>
    <div class="center-text">
    <p>Welcome to the <span class="highlight">Livestock And Plant Disease Predictor System</span>! 🌿🔍</p>
    <p>Our mission is to help in identifying plant and livestock diseases efficiently. Upload an image of a plant or livestock, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!</p>

    <p class="important-text">How It Works</p>
    <ol>
    <li><span class="highlight">Upload Image:</span> Go to the <b>Disease Recognition</b> page and upload an image of a plant or livestock with suspected diseases.</li>
    <li><span class="highlight">Analysis:</span> Our system will process the image using advanced algorithms to identify potential diseases.</li>
    <li><span class="highlight">Results:</span> View the results and recommendations for further action.</li>
    </ol>

    <p class="important-text">Why Choose Us?</p>
    <ul>
    <li><b>Accuracy:</b> Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.</li>
    <li><b>User-Friendly:</b> Simple and intuitive interface for seamless user experience.</li>
    <li><b>Fast and Efficient:</b> Receive results in seconds, allowing for quick decision-making.</li>
    </ul>

    <p class="important-text">Get Started</p>
    <p>Click on the <b>Disease Recognition</b> page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System as well as Livestock Disease detection.</p>

    <p class="important-text">About Us</p>
    <p>Learn more about the project, our team, and our goals on the <b>About</b> page.</p>
    </div>
    """,
    unsafe_allow_html=True
)
    
    #  About Page

elif(app_mode == "About"):
    set_background("#334172 ")
    st.header("About")
    
    # Add an image or banner for visual appeal (if relevant)
    # st.image("path_to_image.png", use_column_width=True)  # Uncomment this line to add a relevant image at the top

    st.markdown("""
        <style>
        .about-section {
            text-align: left;
            font-size: 18px;
            line-height: 1.8;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #39c5bb;
            margin-top: 30px;
        }
        .content-list {
            font-size: 18px;
            margin-top: 10px;
            line-height: 1.6;
        }
        .highlight {
            font-weight: bold;
            color: #e84393;
        }
        </style>
        
        <div class="about-section">
            <p class="section-title">About Dataset</p>
            <p>This dataset is recreated using offline augmentation from the original dataset. 
            The original dataset can be found on this <a href="https://github.com" target="_blank">GitHub repo</a>.
            This dataset consists of about <span class="highlight">87K RGB images</span> of healthy and diseased crop leaves, 
            categorized into <span class="highlight">38 different classes</span>. The dataset is divided into an 
            <span class="highlight">80/20 ratio</span> of training and validation sets, preserving the directory structure.</p>
            <p>A new directory containing <span class="highlight">33 test images</span> is created later for prediction purposes.</p>
            
            <p class="section-title">Content</p>
            <ul class="content-list">
                <li>Train: <span class="highlight">70,295 images</span></li>
                <li>Test: <span class="highlight">33 images</span></li>
                <li>Validation: <span class="highlight">17,572 images</span></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


    #    Disease Recognition

elif(app_mode == "Disease Recognition"):
    set_background("#8d5b76")
    st.header("🌱 Disease Recognition")
    
    # File uploader for the image
    test_image = st.file_uploader("📂 Choose an Image (png, jpg, jpeg):", type=['png', 'jpg', 'jpeg'])
    
    if test_image is not None:
        # Display the image with a caption
        if st.button("Show Image"):
            st.image(test_image, width=400, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()  # Fun effect
            st.write("🔍 **Analyzing Image...**")
            
            # Call the model prediction function and handle the result
            result_index = model_prediction(test_image)
            
            # Class labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbig_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            predicted_class = class_name[result_index]

            # Display result with conditional formatting for healthy/disease
            if "healthy" in predicted_class:
                st.success(f"✅ The plant is **{predicted_class.split('___')[1].replace('_', ' ')}**! Your crop looks healthy! 🎉")
            else:
                st.error(f"⚠️ The plant is suffering from **{predicted_class.split('___')[1].replace('_', ' ')}**. Take immediate action! 🛠️")

            # Provide tips for handling the result
            st.markdown("""
            <style>
            .result-container {
                font-size: 18px;
                margin-top: 30px;
                padding: 10px;
                background-color: #2d8c52;
                border-radius: 8px;
                line-height: 1.6;
            }
            .highlight {
                font-weight: bold;
                color: #e84393;
            }
            .healthy {
                color: green;
            }
            .diseased {
                color: red;
            }
            </style>
            <div class="result-container">
                <p>If the prediction indicates a disease, we recommend you:</p>
                <ul>
                    <li>Inspect nearby plants for similar symptoms.</li>
                    <li>Consult with an agronomist or a plant specialist.</li>
                    <li>Apply appropriate fungicides, herbicides, or other recommended treatments.</li>
                    <li>Consider isolating the affected plants to prevent the spread.</li>
                </ul>
                <p>If your plant is healthy, maintain your current care regimen, but continue monitoring for any unusual changes!</p>
            </div>
            """, unsafe_allow_html=True)



elif app_mode == "Data and Matrix":
    # Create an empty placeholder
    placeholder = st.empty()

    # Show Lottie animation inside the placeholder
    with placeholder:
        st_lottie(lottie_animation, height=200, key="loading_animation")
    
    # Simulate a delay (for example, 2 seconds) before hiding the Lottie animation and loading the content
    time.sleep(2)

    # Clear the placeholder (this will remove the Lottie animation)
    placeholder.empty()

    # Enhanced Header with center alignment and styled
    st.markdown(
        """
        <h2 style='text-align: center; margin-bottom: 30px; color: #4CAF50; font-size: 36px;'>📊 Model Related Details</h2>
        """,
        unsafe_allow_html=True
    )

    # First chart with caption, using st.image for better image loading
    st.markdown("<h4 style='text-align: center; color: #FF5722;'>🔍 Accuracy Results Visualization</h4>", unsafe_allow_html=True)
    st.image("acccuracy_plant.jpeg", caption="Training vs Validation Accuracy Over Epochs", use_column_width=True)

    # Horizontal separator
    st.markdown("<hr style='border: 1px solid #E0E0E0; margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    # Second chart with caption, using st.image for better image loading
    st.markdown("<h4 style='text-align: center; color: #03A9F4;'>📉 Confusion Matrix</h4>", unsafe_allow_html=True)
    st.image("confusion_matrix_Plant.jpeg", caption="Plant Disease Prediction Confusion Matrix", use_column_width=True)

    # Optional: Add some description about the results below the charts, with premium design
    st.markdown("""
        <style>
        .description {
            text-align: center;
            font-size: 20px;
            color: #333;
            padding: 30px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 40px;
        }
        </style>
        <div class="description">
            These visualizations represent the performance of the plant disease prediction model. 
            The accuracy plot shows how the model's accuracy improves over time with training, 
            while the confusion matrix helps us understand the model's performance across different categories.
        </div>
    """, unsafe_allow_html=True)
