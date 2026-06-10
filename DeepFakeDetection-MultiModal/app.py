from flask import Flask, render_template, request
from model.model2 import process_video, process_audio, VideoFeatureExtractor, AudioFeatureExtractor, DeepfakeClassifier
import os
import torch

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models once
video_model = VideoFeatureExtractor().to(DEVICE)
audio_model = AudioFeatureExtractor().to(DEVICE)
classifier = DeepfakeClassifier().to(DEVICE)

# Load trained weights
video_model.load_state_dict(torch.load("video_model.pth", map_location=DEVICE))
audio_model.load_state_dict(torch.load("audio_model.pth", map_location=DEVICE))
classifier.load_state_dict(torch.load("classifier.pth", map_location=DEVICE))

video_model.eval()
audio_model.eval()
classifier.eval()


@app.route('/')
def home():
    return render_template('index.html', content="Upload a video to generate output.")


@app.route('/analyze', methods=['POST'])
def handle_submit():
    if 'video' not in request.files:
        return render_template('index.html', content="No video file received. Please upload a valid video.")
    
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)

    try:
        video_file.save(video_path)

        video_frames = process_video(video_path).to(DEVICE)
        audio_input = process_audio(video_path).to(DEVICE)

        with torch.no_grad():
            video_features = video_model(video_frames)
            audio_features = audio_model(audio_input)
            prediction_tensor = classifier(video_features, audio_features)

            predicted_class = torch.argmax(prediction_tensor, dim=1).item()
            prediction = "Real" if predicted_class == 0 else "Fake"

        return render_template('index.html', content=f"The uploaded video is predicted as: {prediction}")

    except Exception as e:
        return render_template('index.html', content=f"Error processing video: {str(e)}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == '__main__':
    app.run(debug=True)