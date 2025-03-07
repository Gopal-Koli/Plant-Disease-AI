import google.generativeai as genai
from pathlib import Path
import gradio as gr 
from dotenv import load_dotenv  
import os

# test_imports.py
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    import gradio as gr
    print("Imports are successful!")
except ImportError as e:
    print(f"Import error: {e}")


# Load environment variables from a .env file
load_dotenv()

# Configure the GenerativeAI API key using the loaded environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel with the specified model name, configuration, and safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Function to generate a response based on a prompt and an image path
def generate_gemini_response(prompt, image_path):
    image_data = read_image_data(image_path)
    response = model.generate_content([prompt, image_data])
    return response.text

# Initial input prompt for the plant pathologist
input_prompt = """

किसान मित्रों, आपका Farmer Buddy Website पर हार्दिक स्वागत है... हम आपकी फसलों के रोगों के बारे में निश्चित रूप से जानकारी प्रदान करेंगे।

विश्लेषण दिशा-निर्देश:
विश्लेषण मार्गदर्शक सिद्धांत:

रोग की पहचान: पौधों के रोगों को सटीक रूप से पहचानने और विशिष्ट करने के लिए प्रदान की गई जानकारी या नमूनों की जांच करें।
विस्तृत निष्कर्ष: प्रभावित पौधों के भाग, लक्षणों और संभावित कारणों के साथ पहचाने गए पौधों के रोगों के स्वरूप और व्यापकता पर गहरे निष्कर्ष प्रदान करें।
अगले कदम: पहचाने गए पौधों के रोगों के प्रबंधन और नियंत्रण के लिए सुझाए गए उपायों का खाका प्रस्तुत करें। इसमें उपचार के विकल्प, निवारक उपाय या आगे की जांच शामिल हो सकती है।
सिफारिशें: पौधों के स्वास्थ्य को बनाए रखने, रोग के प्रसार को रोकने और पौधों की समग्र भलाई के लिए सलाह दें।
महत्वपूर्ण नोट: एक पौध रोग विशेषज्ञ के रूप में, आपकी अंतर्दृष्टि कृषि और पौधों के प्रबंधन में सूचित निर्णय लेने के लिए महत्वपूर्ण है। आपका उत्तर संपूर्ण, संक्षिप्त और पौधों के स्वास्थ्य पर केंद्रित होना चाहिए।
अस्वीकरण:
"कृपया ध्यान दें कि प्रदान की गई जानकारी पौधों के रोग विज्ञान पर आधारित है और यह व्यावसायिक कृषि परामर्श का स्थान नहीं ले सकती है। कोई भी नीति या उपचार लागू करने से पहले योग्य कृषि विशेषज्ञ से परामर्श लें।"

पौधों के स्वास्थ्य और उत्पादकता को सुनिश्चित करने में आपकी भूमिका महत्वपूर्ण है। प्रदान की गई जानकारी या नमूनों का विश्लेषण करने के लिए आगे बढ़ें, संरचना का पालन करें।

"""
# English prompt

# Function to process uploaded files and generate a response
def process_uploaded_files(files):
    file_path = files[0].name if files else None
    response = generate_gemini_response(input_prompt, file_path) if file_path else None
    return file_path, response

# Gradio interface setup
with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    # Upload button for user to provide images
    upload_button = gr.UploadButton(
        "Click to Upload an Image",
        file_types=["image"],
        file_count="multiple",
    )
     # Set up the upload button to trigger the processing function
    upload_button.upload(process_uploaded_files, upload_button, combined_output)

# Launch the Gradio interface with debug mode enabled
demo.launch(debug=True)