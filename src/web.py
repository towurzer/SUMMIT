from flask import jsonify, Flask, request
from flask_cors import CORS

def create_web_api(config, model):
	api = Flask(__name__)
	CORS(api, origins=["http://localhost:3457", "http://127.0.0.1:3457"]) # CORS to make web app work

	@api.route('/')
	def index():
		return "Hello world! Use GET/POST /translate with parameter text!"

	@api.route('/translate', methods=['GET', 'POST'])
	def translate():
		source = request.args.get('text')
		if source is None:
			source = request.get_data(as_text=True)
			print(source)
		try:
			translated = model.decode(source)

			return translated
		except:
			return "Cannot translate", 400

	return api