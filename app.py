from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model/spam_model_logreg.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        message = request.form["message"]
        pred = model.predict([message])[0]
        prediction = "SPAM" if pred == 1 else "HAM"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

