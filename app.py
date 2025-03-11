import streamlit as st
import os
import random
import string
import numpy as np
import cv2
import math
 # for local Streamlit, we don't need this, can use st.image
from PIL import Image
import google.generativeai as genai

# ------------------------------
# Function to process image (mimicking original Colab cells)
# ------------------------------
def process_image_and_call_gemini(uploaded_file, gemini_api_key):
    first_read_image = None
    image = None
    roi_pil = None
    gemini_response_text = None
    error_message = None

    try:
        # ------------------------------
        # Cell 1: Load the Image
        # ------------------------------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        first_read_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if first_read_image is None:
            error_message = f"Error: Could not load image '{uploaded_file.name}'. Please upload a valid image file."
            return None, None, None, error_message # Early return if image loading fails


        # ------------------------------
        # Cell 2: Convert Image to RGB and Show Basic Info
        # ------------------------------
        # Convert from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(first_read_image, cv2.COLOR_BGR2RGB)
        rows, columns, channels = image.shape
        mean_color = cv2.mean(image)

        # ------------------------------
        # Cell 3: Identify Dark Pixels Across Columns (ROI Determination)
        # ------------------------------
        numbers_of_dark_pixels_in_each_column = [0 for _ in range(columns)]
        addition_of_row_numbers_having_dark_pixels_in_each_column = [0 for _ in range(columns)]
        average_of_row_numbers_having_dark_pixels_in_each_column = [0 for _ in range(columns)]

        # Iterate through each column and row
        for x in range(columns):
            num = 0
            for y in range(rows):
                pixel = image[y, x]
                # Threshold: consider a pixel dark if any channel is less than half its mean value
                if pixel[0] > mean_color[0] * 0.5 and pixel[1] > mean_color[1] * 0.5 and pixel[2] > mean_color[2] * 0.5:
                    continue  # Not dark enough
                else:
                    num += 1
                    addition_of_row_numbers_having_dark_pixels_in_each_column[x] += y
            if num > 2:
                numbers_of_dark_pixels_in_each_column[x] = num
                average_of_row_numbers_having_dark_pixels_in_each_column[x] = addition_of_row_numbers_having_dark_pixels_in_each_column[x] / num

        # ------------------------------
        # Cell 4: Compute Average Dark Pixels and Determine ROI Columns
        # ------------------------------
        average_dark_pixels = 0
        count = 0
        for x in range(columns):
            if numbers_of_dark_pixels_in_each_column[x] > 0:
                average_dark_pixels += numbers_of_dark_pixels_in_each_column[x]
                count += 1

        x_min = -1
        x_max = 0
        if count > 0:
            average_dark_pixels /= count
            for x in range(columns):
                if numbers_of_dark_pixels_in_each_column[x] > average_dark_pixels + 15:
                    if x_min < 0:
                        x_min = x
                    x_max = x

        # ------------------------------
        # Cell 5: Process ROI and Determine y_min, y_max, and Adjust Margins
        # ------------------------------
        if x_min >= 0:
            y_min = average_of_row_numbers_having_dark_pixels_in_each_column[x_min]
            y_max = average_of_row_numbers_having_dark_pixels_in_each_column[x_max]

            # Add horizontal margin
            x_min_2 = max(x_min - 20, 0)
            x_max_2 = min(x_max + 20, len(average_of_row_numbers_having_dark_pixels_in_each_column) - 1)

            # Interpolate to adjust y_min and y_max based on the change across the ROI
            ratio = (average_of_row_numbers_having_dark_pixels_in_each_column[x_max_2] -
                     average_of_row_numbers_having_dark_pixels_in_each_column[x_min_2]) / (x_max_2 - x_min_2)
            y_min = int(ratio * (x_min - x_min_2) + average_of_row_numbers_having_dark_pixels_in_each_column[x_min_2])
            y_max = int(ratio * (x_max - x_max_2) + average_of_row_numbers_having_dark_pixels_in_each_column[x_max_2])
            w = x_max - x_min

            # ------------------------------
            # Cell 6: Extract the ROI, Resize It, and Display
            # ------------------------------
            if w > 10 and -0.7 < ratio < 0.7:
                # Define ROI coordinates with an additional vertical margin
                x1 = min(x_min, x_max)
                x2 = max(x_min, x_max)
                # Choose y coordinates with a ±15 pixel margin; adjust as needed based on your image
                y1 = min(y_min + 15, y_max + 15, y_max - 15, y_min - 15)
                y2 = max(y_min + 15, y_max + 15, y_max - 15, y_min - 15)

                # Resize the ROI while preserving the aspect ratio (forcing height to 50 pixels)
                roi = image[y1:y2, x1:x2]
                res = cv2.resize(roi, (x2 - x1, 50), interpolation=cv2.INTER_NEAREST)

                # Convert the ROI (a NumPy array in RGB) to a PIL Image
                roi_pil = Image.fromarray(res)


                # ------------------------------
                # Cell 7: Convert ROI to PIL Format and Call Gemini-1.5-Flash using google-generativeai
                # ------------------------------
                if roi_pil:

                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')

                    # Create the prompt as in your Streamlit example
                    prompt = (
                        "Analyze this resistor image and determine:\n"
                        "1. Resistance value in ohms (Ω) with proper unit prefix (e.g., Ω, kΩ, MΩ)\n"
                        "2. Tolerance percentage\n"
                        "Provide the answer in this exact format:\n"
                        "Resistance: [value]\n"
                        "Tolerance: [value]%\n"
                        "If uncertain, state 'Cannot determine' for unknown values."
                    )

                    try:
                        # Generate content using both the prompt and the image
                        response = model.generate_content([prompt, roi_pil])
                        if response.text:
                            resistance = "Unknown"
                            tolerance = "Unknown"
                            # Parse the response text line-by-line
                            for line in response.text.split('\n'):
                                if line.lower().startswith('resistance'):
                                    resistance = line.split(':')[-1].strip()
                                elif line.lower().startswith('tolerance'):
                                    tolerance = line.split(':')[-1].strip()
                            gemini_response_text = f"Resistance: {resistance}\nTolerance: {tolerance}"

                        else:
                            error_message = "No text response received from Gemini API."
                    except Exception as e:
                        error_message = f"Error during Gemini API call: {e}"
            else:
                error_message = "ROI could not be properly extracted (conditions not met)."
        else:
            error_message = "ROI could not be properly extracted (x_min < 0)."

    except Exception as main_error:
        error_message = f"An error occurred during image processing: {main_error}"

    return image, roi_pil, gemini_response_text, error_message


# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("Resistor Analyzer with Gemini")
st.write("Upload an image of a resistor and enter your Gemini API key to analyze it.")

gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and gemini_api_key:
    if not gemini_api_key.startswith("AIzaSy"): # Basic API key format validation
        st.error("Invalid Gemini API key format. Please ensure it starts with 'AIzaSy'.")
    else:
        original_image, roi_image_pil, gemini_output, error_msg = process_image_and_call_gemini(uploaded_file, gemini_api_key)

        if error_msg:
            st.error(error_msg)
        else:
            st.subheader("Original Image")
            st.image(original_image, caption="Uploaded Image", use_column_width=True)

            if roi_image_pil:
                st.subheader("Extracted ROI (Resistor)")
                st.image(roi_image_pil, caption="Region of Interest", use_column_width=True)

            if gemini_output:
                st.subheader("Gemini Analysis Result")
                st.write(gemini_output)
elif uploaded_file is None and gemini_api_key:
    st.info("Please upload an image file to analyze.")
elif uploaded_file is not None and not gemini_api_key:
    st.info("Please enter your Gemini API key to analyze the image.")
elif uploaded_file is None and not gemini_api_key:
    st.info("Please upload an image and enter your Gemini API key.")


# ------------------------------
# Detailed Explanation (as a sidebar for better readability)
# ------------------------------
with st.sidebar.expander("Detailed Explanation", expanded=False):
    st.write("""
    **Streamlit Application Structure and Code Explanation:**

    This Streamlit application replicates the functionality of the provided Python code snippet, which is designed to:

    1. **Upload Image:** Allow users to upload an image of a resistor.
    2. **Process Image (OpenCV):**
        - Load the image using OpenCV.
        - Convert the image from BGR to RGB color space.
        - Identify a Region of Interest (ROI) likely containing the resistor by:
            - Detecting 'dark' pixels in each column.
            - Calculating average dark pixel count.
            - Finding columns with significantly higher dark pixel counts to define the horizontal ROI boundaries (x_min, x_max).
            - Estimating vertical ROI boundaries (y_min, y_max) based on the vertical position of dark pixels within the horizontal ROI.
        - Extract and resize the ROI to a fixed height for consistent Gemini API input.
    3. **Gemini API Call:**
        - Use the Gemini-1.5-Flash model (via the `google-generativeai` library) to analyze the extracted ROI.
        - Prompt Gemini to identify the resistance value and tolerance of the resistor.
    4. **Display Results:**
        - Show the original uploaded image.
        - Display the extracted ROI image.
        - Present the analysis result from Gemini (Resistance and Tolerance values).
        - Display error messages if any issues occur during processing or API calls.

    **Code Structure:**

    - **Import Libraries:** Imports necessary libraries (`streamlit`, `os`, `cv2`, `numpy`, `PIL`, `google.generativeai`).
    - **`process_image_and_call_gemini(uploaded_file, gemini_api_key)` function:**
        - Encapsulates the core image processing and Gemini API call logic, mirroring the original Colab cells.
        - Takes the uploaded file and Gemini API key as input.
        - Returns the original image (for display), ROI PIL image (for display), Gemini output text, and any error messages.
        - Includes error handling within a `try...except` block for robustness.
    - **Streamlit App Layout (`if __name__ == '__main__':`)**:
        - Sets up the Streamlit user interface:
            - Title and description.
            - Input field for Gemini API key (password type for security).
            - File uploader for images.
            - Conditional logic to process the image and display results only when an image is uploaded and API key is provided and valid.
            - Displays original image, ROI, and Gemini output using `st.image` and `st.write`.
            - Shows error messages using `st.error` and info messages using `st.info` for user guidance.
    - **Detailed Explanation Sidebar:**
        - Uses `st.sidebar.expander` to create an expandable sidebar section containing this detailed explanation for better user experience.

    **Mapping to Original Code Functionality:**

    The Streamlit application directly translates the Colab code cells into a functional web application. The `process_image_and_call_gemini` function is structured to mimic the sequential execution of the original cells (Cell 1 to Cell 7).  Each step in the original code is implemented using the same logic and libraries within this function. The Streamlit UI provides the user interface to upload images and input the API key, replacing the Colab-specific file upload and hardcoded API key methods.

    **Key Design Decisions:**

    - **Function Encapsulation:**  The core logic is placed within a function (`process_image_and_call_gemini`) for better code organization, reusability, and testability.
    - **Error Handling:**  `try...except` blocks are used throughout the processing function and in the Streamlit app to catch potential errors and provide informative messages to the user.
    - **Clear UI Structure:** Streamlit's simple API is used to create a user-friendly interface with clear sections for image upload, API key input, image display, and results.
    - **Sidebar Explanation:** A sidebar is used to provide a detailed explanation without cluttering the main application interface.
    - **API Key Security:** The API key input is set to `type="password"` in Streamlit to mask the input for basic security in a public demonstration. **Note:** This is not a secure way to handle API keys in production applications; consider using environment variables or secure key management practices for production.
    - **Basic API Key Validation:** A simple check (`startswith("AIzaSy")`) is added to provide immediate feedback on potentially invalid API keys.

    **Further Improvements (Beyond the Scope of this Task):**

    - **Parameter Tuning:** Allow users to adjust parameters like dark pixel threshold or ROI margins through sliders to fine-tune the ROI extraction process.
    - **Loading Indicator:** Add a loading spinner or progress bar during image processing and Gemini API calls for better user feedback.
    - **More Robust Error Handling:** Implement more specific error handling for different OpenCV and Gemini API errors.
    - **File Download:** Allow users to download the processed ROI image.
    - **UI Enhancements:** Improve the visual design and layout of the Streamlit application for a more polished user experience.
    """)
