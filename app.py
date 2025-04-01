from flask import Flask, request, render_template
import os
from src.predict import correct_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", error="No file selected")
            
        img = request.files["image"]
        if img.filename == '':
            return render_template("index.html", error="No file selected")
            
        blindness_type = request.form.get("type", "protanopia")
        
        # Save input
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
        img.save(input_path)
        
        # Process image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
        try:
            correct_image(input_path, output_path, blindness_type)
            return render_template("result.html", 
                                input_img="input.jpg",
                                output_img="output.jpg",
                                type=blindness_type.upper())
        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    app.run(debug=True)