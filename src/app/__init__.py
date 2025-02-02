from flask import jsonify, Flask, request

def create_app(config, model):
	app = Flask(__name__)

	@app.route('/')
	def index():
		return "Hello world! Use GET/POST /translate with parameter text!"

	@app.route('/translate', methods=['GET', 'POST'])
	def translate():

		source = request.args.get('text')
		if source is None:
			source = request.form.get('text')
		
		try:
			translated = model.decode(source)

			return jsonify({"translated": f"\"{translated}\""})
		except:
			return "Cannot translate", 400

	return app