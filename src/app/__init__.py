from flask import jsonify, Flask, request

def create_app(config, model):
	app = Flask(__name__)

	@app.route('/')
	def index():
		return f"Debug mode is {'on' if app.config["DEBUG"] else 'off'}"

	@app.route('/translate', methods=['GET'])
	def translate():
		

		source = request.args.get('text')
		
		try:

			translated = model.decode(config, source)


			return jsonify({"translated": f"\"{translated}\""})
		except:
			return "Cannot translate", 400

	return app