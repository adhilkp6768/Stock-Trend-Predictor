# Add this route to your existing Flask app
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"}), 200