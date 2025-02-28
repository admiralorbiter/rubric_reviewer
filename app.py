from flask import Flask, request, jsonify, render_template, session
import requests
import base64
import os
import json
import io
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

@app.route('/')
def index():
    return render_template('index.html')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def pdf_to_images(pdf_file):
    """Convert PDF pages to base64-encoded images."""
    images = []
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        encoded_img = base64.b64encode(img_byte_arr).decode('ascii')
        
        images.append(encoded_img)
    
    return images

@app.route('/upload', methods=['POST'])
def upload():
    # Get uploaded PDF files
    rubric_file = request.files.get('rubric_pdf')
    work_file = request.files.get('work_pdf')

    if not rubric_file or not work_file:
        return jsonify({"error": "Both rubric and student work PDFs are required"}), 400

    # Check if files are PDFs
    if not rubric_file.filename.lower().endswith('.pdf') or not work_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Both files must be in PDF format"}), 400

    try:
        # Method 1: Extract text from PDFs
        rubric_text = extract_text_from_pdf(rubric_file)
        
        # Reset file pointer for image conversion
        rubric_file.seek(0)
        work_file.seek(0)
        
        # Method 2: Convert PDFs to images for visual analysis
        rubric_images = pdf_to_images(rubric_file)
        work_images = pdf_to_images(work_file)
        
        # Prepare content for the API
        content = []
        
        # Add system message
        system_message = {
            "role": "system",
            "content": "You are an expert educator specializing in assessment. Evaluate student work against provided rubrics with detailed, constructive feedback."
        }
        
        # Prepare user content with images
        user_content = []
        
        # Add text instruction
        user_content.append({
            "type": "text",
            "text": f"I need to evaluate student work based on a rubric. The rubric text is as follows:\n\n{rubric_text}\n\nBelow are images of both the rubric and student work for visual reference. Please evaluate the student work against each criterion in the rubric. Provide specific feedback on strengths and areas for improvement for each criterion."
        })
        
        # Add rubric images
        for idx, img in enumerate(rubric_images):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                    "detail": "high"
                }
            })
        
        # Add separator
        user_content.append({
            "type": "text", 
            "text": "Above are images of the rubric. Below are images of the student work:"
        })
        
        # Add work images
        for img in work_images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                    "detail": "high"
                }
            })
        
        # Add final instruction
        user_content.append({
            "type": "text",
            "text": """
            Please evaluate the student work against each criterion in the rubric. 
            Format your response as JSON with the following structure:
            {
              "feedback_steps": [
                {
                  "title": "Section title or criterion name",
                  "feedback": "Detailed feedback for this section"
                },
                // More sections...
              ]
            }
            
            Be thorough and constructive in your feedback.
            """
        })
        
        # Prepare the payload for OpenAI API
        payload = {
            "model": "gpt-4o",
            "messages": [
                system_message,
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 1500
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make the API call
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        
        if not response.ok:
            app.logger.error(f"OpenAI API error: {response.text}")
            return jsonify({"error": "Failed to get feedback from OpenAI API"}), 500

        # Extract the content from the API response
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        
        # Try to parse the JSON from the content
        try:
            # Extract JSON if it's embedded in markdown or other text
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
                feedback_data = json.loads(json_content)
            else:
                feedback_data = json.loads(content)
                
            steps = feedback_data.get("feedback_steps", [])
            
            # Store steps in session for potential step-by-step retrieval
            session['feedback_steps'] = steps
            session['current_step'] = 0
            
            # Return the first step and total count
            return render_template('feedback.html', 
                                  step=steps[0], 
                                  current=1, 
                                  total=len(steps))
            
        except json.JSONDecodeError:
            app.logger.error(f"Failed to parse JSON from API response: {content}")
            return jsonify({"error": "Failed to parse feedback from API"}), 500
    
    except Exception as e:
        app.logger.error(f"Error processing PDFs or calling OpenAI API: {str(e)}")
        return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

@app.route('/feedback/step', methods=['POST'])
def feedback_step():
    # Get the requested step (next or previous)
    direction = request.form.get('direction', 'next')
    
    # Get stored steps from session
    steps = session.get('feedback_steps', [])
    current = session.get('current_step', 0)
    
    if not steps:
        return jsonify({"error": "No feedback steps available"}), 400
    
    # Calculate the new step index
    if direction == 'next':
        new_index = min(current + 1, len(steps) - 1)
    else:  # previous
        new_index = max(current - 1, 0)
    
    # Update the session
    session['current_step'] = new_index
    
    # Return the full feedback template
    return render_template('feedback.html', 
                         step=steps[new_index], 
                         current=new_index + 1, 
                         total=len(steps))

if __name__ == '__main__':
    app.run(debug=True)