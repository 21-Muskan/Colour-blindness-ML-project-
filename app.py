from flask import Flask, request, render_template, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
from src.predict import correct_image
from src.simulate import simulate_color_blindness
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-123'  # Required for flash messages

# Configuration
UPLOAD_FOLDER = 'static/uploads'
SIMULATED_FOLDER = 'static/simulated'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SIMULATED_FOLDER'] = SIMULATED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SIMULATED_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if file was uploaded
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['image']
        
        # If user doesn't select file
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # Get color blindness type
                blindness_type = request.form.get("type", "protanopia").lower()
                valid_types = ["protanopia", "deuteranopia", "tritanopia"]
                
                if blindness_type not in valid_types:
                    flash('Invalid color blindness type selected', 'error')
                    return redirect(request.url)
                
                # Generate unique filenames
                timestamp = str(int(time.time()))
                filename = secure_filename(file.filename)
                base_name, ext = os.path.splitext(filename)
                
                input_filename = f"input_{timestamp}{ext}"
                output_filename = f"output_{timestamp}{ext}"
                simulated_filename = f"simulated_{timestamp}{ext}"
                
                # Create full paths
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                simulated_path = os.path.join(app.config['SIMULATED_FOLDER'], simulated_filename)
                
                # Save original file
                file.save(input_path)
                
                # Verify image was saved
                if not os.path.exists(input_path):
                    flash('Failed to save uploaded image', 'error')
                    return redirect(request.url)
                
                # Process the image
                simulate_color_blindness(input_path, simulated_path, blindness_type)
                correct_image(input_path, output_path, blindness_type)
                
                # Verify outputs were created
                if not all(os.path.exists(p) for p in [simulated_path, output_path]):
                    flash('Image processing failed - output files not created', 'error')
                    return redirect(request.url)
                
                # Prepare display names (without 'static/' prefix)
                result_data = {
                    'input_img': f"uploads/{input_filename}",
                    'simulated_img': f"simulated/{simulated_filename}",
                    'output_img': f"uploads/{output_filename}",
                    'type': blindness_type.capitalize()
                }
                
                return render_template("result.html", **result_data)
                
            except Exception as e:
                # Clean up any created files if error occurs
                for filepath in [input_path, output_path, simulated_path]:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        
        else:
            flash('Allowed file types are: png, jpg, jpeg, webp', 'error')
            return redirect(request.url)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)