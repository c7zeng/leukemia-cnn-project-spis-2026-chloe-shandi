from flask import Flask, render_template, request, send_from_directory
from model import test_model
import os

app = Flask(__name__)

DEMO_FOLDER = "data/demo-images"


# =========================
# HOME / MODEL DEMO
# =========================

@app.route("/")
def home():

    images = os.listdir(DEMO_FOLDER)

    return render_template(
        "index.html",
        images=images
    )


# =========================
# ANALYZE IMAGE
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():

    filename = request.form["image"]

    file_path = os.path.join(DEMO_FOLDER, filename)

    result, confidence, ground_truth = test_model(file_path)

    return render_template(
        "index.html",
        images=os.listdir(DEMO_FOLDER),
        selected_image=filename,
        result=result,
        confidence=confidence,
        ground_truth=ground_truth
    )


# =========================
# SERVE DEMO IMAGES
# =========================

@app.route("/demo-images/<filename>")
def demo_image(filename):

    return send_from_directory(
        DEMO_FOLDER,
        filename
    )


# =========================
# LAB WRITEUP
# =========================

@app.route("/writeup")
def writeup():

    return render_template("lab.html")


# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )