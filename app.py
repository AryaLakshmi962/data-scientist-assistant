from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os, uuid, shutil
from werkzeug.utils import secure_filename
from model.data_scientist_assistant import run_pipeline

app = Flask(__name__)
app.secret_key = "change_this_in_production"
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
OUTPUT_FOLDER = os.path.join(app.root_path, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXT = {'.csv'}

def allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=['POST'])
def analyze():
    if 'dataset' not in request.files:
        flash("No file part")
        return redirect(url_for('index'))
    file = request.files['dataset']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    if not allowed(file.filename):
        flash("Only CSV files are allowed")
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{filename}")
    file.save(saved_path)
    out_dir = os.path.join(OUTPUT_FOLDER, uid)
    os.makedirs(out_dir, exist_ok=True)
    # run pipeline
    report = run_pipeline(saved_path, out_dir)
    # zip output dir
    zip_path = shutil.make_archive(out_dir, 'zip', out_dir)
    zip_name = os.path.basename(zip_path)
    return render_template("result.html", report=report, zip_name=zip_name)

@app.route("/download/<zipname>")
def download(zipname):
    zip_path = os.path.join(OUTPUT_FOLDER, zipname)
    if not os.path.exists(zip_path):
        return "Not found", 404
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
