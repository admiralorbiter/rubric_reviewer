from flask import Flask, request, jsonify, render_template, session
import requests
import base64
import os
import json
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

@app.route('/upload', methods=['POST'])
def upload():
    # Get uploaded image and rubric text
    image_file = request.files.get('image')
    rubric_text = request.form.get('rubric')

    if not rubric_text or not image_file:
        return jsonify({"error": "Rubric text or image is missing"}), 400

    # Check if file is an allowed image type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid image format. Please upload PNG, JPG, JPEG, or GIF."}), 400

    # Convert image to base64
    image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create prompt for OpenAI API
    prompt = f"""
    Analyze the following image according to this rubric:
    
    {rubric_text}
    
    Provide detailed feedback for each section of the rubric. Format your response as JSON with the following structure:
    {{
      "feedback_steps": [
        {{
          "title": "Section title or criterion name",
          "feedback": "Detailed feedback for this section"
        }},
        // More sections...
      ]
    }}
    
    Be thorough and constructive in your feedback.
    """
    
    # Prepare the payload for OpenAI API
    payload = {
        "model": "gpt-4-vision-preview",  # Or another appropriate model
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
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
        app.logger.error(f"Error calling OpenAI API: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

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
    
    # Return the requested step
    return render_template('feedback_step.html', 
                          step=steps[new_index], 
                          current=new_index + 1, 
                          total=len(steps))

if __name__ == '__main__':
    app.run(debug=True)