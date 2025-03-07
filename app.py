from flask import Flask, render_template, request, redirect, url_for, send_file
import torch
import re
import pdfkit
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import io
import requests
import os


app = Flask(__name__)


# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model parameters for caption generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# List of common negation words
NEGATION_WORDS = {"not", "never", "no", "none", "nothing", "nobody", "nowhere", "neither"}

def predict_caption(image):
    if image is not None:
        try:
            if isinstance(image, str) and image.startswith("http"):  # Handle URL images
                response = requests.get(image, stream=True)
                if response.status_code == 200:
                    i_image = Image.open(io.BytesIO(response.content))
                else:
                    return None, "Failed to download image."
            else:
                i_image = Image.open(image)

            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

        except Exception as e:
            return None, f"Error processing image: {e}"

        inputs = processor(images=i_image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return generated_caption.strip(), None

    return None, "No image provided."

def adjust_similarity(caption, similarity_score):
    """Adjusts similarity score if negation words are present in the caption."""
    words = set(re.findall(r'\b\w+\b', caption.lower()))
    
    if words & NEGATION_WORDS:  # If negation words are found
        similarity_score *= 0.5  # Reduce score (adjust factor if needed)
    
    return round(similarity_score, 4)

@app.route('/')
def welcome():
    return render_template('main.html')

@app.route('/similarity_checker')
def similarity_checker():
    return render_template('index.html')

def generate_heatmap(generated_caption, user_caption):
    """Generate and save a heatmap showing word similarity."""
    
    # Ensure 'static' directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)  # Create the directory if it does not exist

    heatmap_path = os.path.join(static_dir, "heatmap.png")

    # Skip generation if either caption is empty
    if not generated_caption or not user_caption:
        return heatmap_path if os.path.exists(heatmap_path) else None

    # Tokenize captions
    gen_tokens = generated_caption.split()
    user_tokens = user_caption.split()

    # Encode each word
    gen_embeddings = np.array([caption_model.encode(word) for word in gen_tokens]) if gen_tokens else np.zeros((1, 384))
    user_embeddings = np.array([caption_model.encode(word) for word in user_tokens]) if user_tokens else np.zeros((1, 384))

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(gen_embeddings, user_embeddings.T)

    # Normalize to avoid division errors
    norm_gen = np.linalg.norm(gen_embeddings, axis=1, keepdims=True) + 1e-8
    norm_user = np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8
    similarity_matrix /= (norm_gen * norm_user.T)

    # Create and save heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=user_tokens, yticklabels=gen_tokens, cmap="coolwarm")
    plt.xlabel("User Caption Words")
    plt.ylabel("Generated Caption Words")
    plt.title("Word Similarity Heatmap")
    
    plt.savefig(heatmap_path)  # Save the heatmap
    plt.close()

    return heatmap_path  # Return the file path

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    user_caption = request.form['caption']

    uploaded_file = request.files.get('image')
    if not uploaded_file:
        return render_template('index.html', error="Please upload an image.")

    print("Received request:", request.files)
    print("Uploaded file:", uploaded_file.filename)

    generated_caption, error_message = predict_caption(uploaded_file)

    if error_message:
        return render_template('index.html', error=error_message)

    embeddings1 = caption_model.encode([generated_caption])
    embeddings2 = caption_model.encode([user_caption])
    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings1), torch.tensor(embeddings2)
    ).item()

    adjusted_similarity = adjust_similarity(user_caption, similarity)
    result = "Non Out of Context Media" if adjusted_similarity > 0.5 else "Out of Context Media"

    heatmap_path = generate_heatmap(generated_caption, user_caption)

    return render_template('result.html', 
                          generated_caption=generated_caption, 
                          user_caption=user_caption,
                          similarity_score=round(similarity, 4), 
                          result="Out of Context" if similarity < 0.5 else "Non Out of Context",
                          heatmap_path=url_for('static', filename='heatmap.png'))

@app.route('/report', methods=['GET'])
def show_report():
    """
    Render the HTML report page with detection details.
    """
    generated_caption = request.args.get('generated_caption')
    user_caption = request.args.get('user_caption')
    similarity_score = request.args.get('similarity_score')
    result = request.args.get('result')

    return render_template('report.html', 
                          generated_caption=generated_caption,
                          user_caption=user_caption,
                          similarity_score=similarity_score, 
                          result=result)

@app.route('/download_report', methods=['GET', 'POST'])
def download_report():
    """
    Generate a PDF report including the heatmap and send it as a file download.
    """
    try:
        # Handle both GET and POST requests
        if request.method == 'POST':
            generated_caption = request.form.get('generated_caption', 'N/A')
            similarity_score = request.form.get('similarity_score', 'N/A')
            result = request.form.get('result', 'N/A')
            user_caption = request.form.get('user_caption', 'N/A')  # Get user-provided caption
        else:  # GET request
            generated_caption = request.args.get('generated_caption', 'N/A')
            similarity_score = request.args.get('similarity_score', 'N/A')
            result = request.args.get('result', 'N/A')
            user_caption = request.args.get('user_caption', 'N/A')

        # Generate the heatmap before rendering the report
        if generated_caption != 'N/A' and user_caption != 'N/A':
            heatmap_path = generate_heatmap(generated_caption, user_caption)
            # Get absolute path for the heatmap
            abs_heatmap_path = os.path.join(os.getcwd(), heatmap_path)
            print(f"Generated heatmap at: {abs_heatmap_path}")
        else:
            abs_heatmap_path = None

        # Render the HTML report as a string
        report_html = render_template('report.html', 
                                     generated_caption=generated_caption,
                                     user_caption=user_caption,
                                     similarity_score=similarity_score, 
                                     result=result,
                                     heatmap_path=abs_heatmap_path)

        # Create a full path for the PDF in a writable directory
        app_dir = os.getcwd()
        pdf_path = os.path.join(app_dir, 'report.pdf')
        print(f"Attempting to save PDF to: {pdf_path}")
        
        # Create PDF with options to handle local image paths
        options = {
            'quiet': '',
            'enable-local-file-access': '',  # Important for loading local images
            'encoding': "UTF-8",
        }
        
        # Define configuration for pdfkit
        try:
            # Try to detect wkhtmltopdf installation automatically
            pdfkit_config = pdfkit.configuration()
        except Exception as e:
            print(f"Automatic wkhtmltopdf detection failed: {e}")
            # Fall back to your specified path
            wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            if os.path.exists(wkhtmltopdf_path):
                print(f"Using wkhtmltopdf from: {wkhtmltopdf_path}")
                pdfkit_config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
            else:
                print(f"WARNING: wkhtmltopdf not found at {wkhtmltopdf_path}")
                # Try with just the executable name as a last resort
                pdfkit_config = pdfkit.configuration(wkhtmltopdf='wkhtmltopdf')
        
        try:
            pdfkit.from_string(report_html, pdf_path, options=options, configuration=pdfkit_config)
        except Exception as e:
            print(f"PDF generation failed: {e}")
            try:
                # Try without configuration as a fallback
                pdfkit.from_string(report_html, pdf_path, options=options)
            except Exception as e2:
                print(f"Second PDF generation attempt failed: {e2}")
                return f"Error generating PDF: {str(e2)}", 500
        
        # Check if the file was created
        if os.path.exists(pdf_path):
            print(f"PDF file created successfully at {pdf_path}")
            return send_file(pdf_path, as_attachment=True, download_name='detection_report.pdf')
        else:
            print("PDF file was not created despite no exceptions")
            return "Error: PDF could not be generated", 500
            
    except Exception as e:
        print(f"Unexpected error in download_report: {str(e)}")
        return f"Error generating PDF: {str(e)}", 500


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_data = {
        "reaction": request.form.get('reaction'),
        "generated_caption": request.form.get('generated_caption'),
        "similarity_score": request.form.get('adjusted_similarity'),
        "result": request.form.get('result'),
    }
    print(feedback_data)

    return render_template('thank_you.html', feedback_data=feedback_data)

if __name__ == '__main__':
    # Print debug information at startup
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {os.path.dirname(os.__file__)}")
    
    # Try to locate wkhtmltopdf
    try:
        import subprocess
        result = subprocess.run(['where', 'wkhtmltopdf'] if os.name == 'nt' else ['which', 'wkhtmltopdf'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"wkhtmltopdf found at: {result.stdout.strip()}")
        else:
            print("wkhtmltopdf not found in PATH")
    except Exception as e:
        print(f"Error checking wkhtmltopdf: {e}")
    
    app.run(debug=True)