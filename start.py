from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
	file = request.files['file']
	print(file.filename)
	in_memory_file = io.BytesIO()
	file.save(in_memory_file)
	data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
	color_image_flag = 1
	img = cv2.imdecode(data, color_image_flag)

	result = DeepFace.analyze(img, actions = ['emotion'])

	return jsonify(result), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)
