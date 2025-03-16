import google.generativeai as genai
from pathlib import Path
import gradio as gr 
from dotenv import load_dotenv  
import os

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

# Initialize the GenerativeModel
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


# Prompts in different languages
prompts = {
    "English": """
Farmer friends, a warm welcome to the Farmer Buddy Website... We will certainly provide information about the diseases affecting your crops.

Analysis Guidelines:
Analytical Guiding Principles:

1. Disease Identification: Examine the provided information or samples to accurately identify and specify plant diseases.
2. Detailed Findings: Offer deep insights into the nature and prevalence of identified plant diseases, including affected plant parts, symptoms, and possible causes.
3. Next Steps: Present a framework of suggested measures for managing and controlling the identified plant diseases. This may include treatment options, preventive measures, or further investigation.
4. Recommendations: Provide advice to maintain plant health, prevent the spread of diseases, and ensure the overall well-being of plants.

Important Note: As a plant disease expert, your insights are crucial for making informed decisions in agriculture and plant management. Your response should be comprehensive, concise, and focused on plant health.

Disclaimer:
"Please note that the information provided is based on plant pathology and cannot replace professional agricultural consultation. Always consult a qualified agricultural expert before implementing any policy or treatment."
    """,

    "Hindi": """
किसान मित्रों, आपका Farmer Buddy Website पर हार्दिक स्वागत है... हम आपकी फसलों के रोगों के बारे में निश्चित रूप से जानकारी प्रदान करेंगे।

विश्लेषण दिशा-निर्देश:
विश्लेषण मार्गदर्शक सिद्धांत:

1. रोग की पहचान: पौधों के रोगों को सटीक रूप से पहचानने और विशिष्ट करने के लिए प्रदान की गई जानकारी या नमूनों की जांच करें।
2. विस्तृत निष्कर्ष: प्रभावित पौधों के भाग, लक्षणों और संभावित कारणों के साथ पहचाने गए पौधों के रोगों के स्वरूप और व्यापकता पर गहरे निष्कर्ष प्रदान करें।
3. अगले कदम: पहचाने गए पौधों के रोगों के प्रबंधन और नियंत्रण के लिए सुझाए गए उपायों का खाका प्रस्तुत करें। इसमें उपचार के विकल्प, निवारक उपाय या आगे की जांच शामिल हो सकती है।
4. सिफारिशें: पौधों के स्वास्थ्य को बनाए रखने, रोग के प्रसार को रोकने और पौधों की समग्र भलाई के लिए सलाह दें।

महत्वपूर्ण नोट: एक पौध रोग विशेषज्ञ के रूप में, आपकी अंतर्दृष्टि कृषि और पौधों के प्रबंधन में सूचित निर्णय लेने के लिए महत्वपूर्ण है। आपका उत्तर संपूर्ण, संक्षिप्त और पौधों के स्वास्थ्य पर केंद्रित होना चाहिए।

अस्वीकरण:
"कृपया ध्यान दें कि प्रदान की गई जानकारी पौधों के रोग विज्ञान पर आधारित है और यह व्यावसायिक कृषि परामर्श का स्थान नहीं ले सकती है। कोई भी नीति या उपचार लागू करने से पहले योग्य कृषि विशेषज्ञ से परामर्श लें।"
    """,

    "Marathi": """
शेतकरी मित्रांनो, Farmer Buddy Website वर तुमचे हार्दिक स्वागत आहे... आम्ही तुमच्या पिकांवरील रोगांबद्दल निश्चितच माहिती प्रदान करू.

विश्लेषण मार्गदर्शक:
विश्लेषणासाठी मार्गदर्शक तत्त्वे:

1. रोगाची ओळख: दिलेल्या माहितीस किंवा नमुन्यांच्या आधारे पिकांवरील रोग अचूकपणे ओळखून त्यांचे विश्लेषण करा.
2. सविस्तर निष्कर्ष: पिकांवरील रोगांचा स्वरूप व प्रसार यावर सखोल निरीक्षणे द्या, ज्यामध्ये बाधित भाग, लक्षणे, आणि संभाव्य कारणे समाविष्ट असतील.
3. पुढील पावले: ओळखलेल्या रोगांवर नियंत्रण व व्यवस्थापनासाठी उपाययोजना सुचवा. यामध्ये उपचारांचे पर्याय, प्रतिबंधात्मक उपाय किंवा अधिक तपासणीचा समावेश असू शकतो.
4. शिफारसी: पिकांचे आरोग्य टिकवून ठेवणे, रोगांचा प्रसार टाळणे, आणि पिकांची एकूण उत्पादकता सुधारण्यासाठी योग्य सल्ला द्या.

महत्त्वाची टीप: पिकांच्या रोगतज्ज्ञ (Plant Disease Expert) म्हणून, शेती व्यवस्थापनासाठी तुमचे मार्गदर्शन फार महत्त्वाचे आहे. तुमचे उत्तर सविस्तर, नेमके व पिकांच्या आरोग्यावर केंद्रित असावे.

अस्वीकरण:
"कृपया लक्षात घ्या की दिलेली माहिती पिकांच्या रोगशास्त्रावर आधारित आहे व ती व्यावसायिक शेती सल्ल्याचा पर्याय ठरू शकत नाही. कोणतीही धोरणे किंवा उपचार अमलात आणण्यापूर्वी पात्र कृषी तज्ज्ञांचा सल्ला घ्या."
    """,
}

# Function to process uploaded files and generate a response
def process_uploaded_files(files, language):
    file_path = files[0].name if files else None
    prompt = prompts.get(language, prompts["English"])
    response = generate_gemini_response(prompt, file_path) if file_path else None
    return file_path, response

# Gradio interface setup
with gr.Blocks() as demo:
    file_output = gr.Textbox(label="File Path")
    response_output = gr.Textbox(label="Response")
    image_output = gr.Image(label="Uploaded Image")

    # Language selection buttons
    language_selector = gr.Radio(choices=["English", "Hindi", "Marathi"], value="English", label="Select Language")

    # Upload button for user to provide images
    upload_button = gr.UploadButton("Click to Upload an Image", file_types=["image"], file_count="multiple")

    # Set up the upload button to trigger the processing function
    upload_button.upload(
        lambda files, lang: (files[0].name if files else None, generate_gemini_response(prompts.get(lang, prompts["English"]), files[0].name if files else None), files[0].name if files else None),
        inputs=[upload_button, language_selector],
        outputs=[file_output, response_output, image_output],
    )

# Launch the Gradio interface with debug mode enabled
demo.launch(debug=True)
