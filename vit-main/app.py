import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k', num_labels=4
)

# Path to custom trained weights
model_path = r'C:\Users\Dhinesh kumar A\Downloads\vit-main\vit-main\vit_multiple_sclerosis_final.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Class mapping
class_mapping = {
    0: 'Control-Axial',
    1: 'Control-Sagittal',
    2: 'MS-Axial',
    3: 'MS-Sagittal'
}

# Upload folder
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess image
        image = Image.open(filepath).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        # Inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = class_mapping[predicted_class_idx]

        # Serve uploaded image via /static/uploads
        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            'result.html',
            prediction=predicted_class,
            uploaded_image=image_url
        )

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
