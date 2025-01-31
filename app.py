import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import base64

# Function to set the background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background("D:/Face_Recognition Mini Proj/image2.jpeg")

# Get the current timestamp and formatted date
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Auto-refresh every 2000 ms (2 seconds), with a limit of 100 refreshes
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# Read the CSV file and process attendance
try:
    df = pd.read_csv(f"Attendance/Attendance_{date}.csv")
    st.dataframe(df.style.highlight_max(axis=0))

    # Ensure that each unique name is counted only once
    unique_names_df = df.drop_duplicates(subset='NAME')
    attendance_count = unique_names_df.shape[0]  # Count the number of unique rows

    st.markdown(f"<h3 style='color: white;'>Attendance Count: {attendance_count}</h3>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<h3 style='color: white;'>Attendance file not found for today.</h3>", unsafe_allow_html=True)
except KeyError:
    st.markdown("<h3 style='color: white;'>The 'NAME' column is missing from the attendance file.</h3>", unsafe_allow_html=True)

# Add an image on the web page with white font color for the caption
st.markdown(
    """
    <style>
    .image-caption {
        color: white;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("D:/Face_Recognition Mini Proj/image1.jpeg", caption="Attendance Tracker", use_column_width=True, output_format='JPEG')

# Apply the white font color to the caption
st.markdown("<style>.caption { color: white !important; }</style>", unsafe_allow_html=True)
