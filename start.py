from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
	if 'file' not in request.files:
		return jsonify({'error': 'No file part in the request'}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({'error': 'No selected file'}), 400
	if not file or file.filename.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
		return jsonify({'error': 'Invalid file type'}), 400

	in_memory_file = io.BytesIO()
	file.save(in_memory_file)
	data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
	color_image_flag = 1
	img = cv2.imdecode(data, color_image_flag)

	try:
		result = DeepFace.analyze(img, actions=['emotion'])
	except ValueError:
		return jsonify({'error': 'No face detected in the image'}), 400

	return jsonify(result), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)
