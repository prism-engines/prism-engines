from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/run", methods=["POST"])
def run_prism():
    payload = request.json
    print("Received PRISM Command:", payload)

    # Placeholder until wired into full PRISM engine
    return jsonify({
        "status": "ok",
        "message": "PRISM controller received your request.",
        "payload_received": payload
    })

if __name__ == "__main__":
    app.run(debug=True)
