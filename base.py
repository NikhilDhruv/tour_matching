from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory
from celery_app import generate_embeddings_task, delete_files
import os
import pandas as pd
from flask import Flask, request, send_file
import pandas as pd
from gpt_utils import generate_match_explanation, append_match_explanations



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app = Flask(__name__)

@app.route('/process-matches', methods=['POST'])
def process_matches():
    file = request.files['file']  # Upload CSV file
    matches_df = pd.read_csv(file)

    # Generate GPT explanations
    updated_df = append_match_explanations(matches_df)

    # Save the updated CSV
    output_path = "updated_matches.csv"
    updated_df.to_csv(output_path, index=False)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/match_students", methods=["POST"])
def match_students():
    prospective_file = request.files.get("prospective_students_file")
    current_file = request.files.get("current_students_file")
    
    if not prospective_file or not current_file:
        return "Both files are required!", 400

    prospective_path = os.path.join(app.config['UPLOAD_FOLDER'], "prospective_students.csv")
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_students.csv")

    # Save the uploaded files
    prospective_file.save(prospective_path)
    current_file.save(current_path)
    
    # Pass file paths to the Celery task
    task = generate_embeddings_task.delay(prospective_path, current_path)
    delete_files.apply_async(args=[[prospective_path, current_path]], countdown=3600)
    print("Task ID:", task.id)

    return render_template("loading.html", task_id=task.id)


@app.route("/task_status/<task_id>")
def task_status(task_id):
    print("check 1")
    task = generate_embeddings_task.AsyncResult(task_id)
    print("Task ID:", task_id)
    print("Task State:", task.state)
    if task.state == 'SUCCESS':
        csv_path = task.result['csv_path']
        delete_files.apply_async(args=[[csv_path]], countdown=3600)
        filename = os.path.basename(csv_path)
        # Return the URL to the results page as JSON
        return jsonify({"status": "SUCCESS", "redirect_url": url_for('results', filename=filename)})
    elif task.state == 'FAILURE':
        return jsonify({"status": "FAILURE", "error": str(task.info)}), 500
    else:
        return jsonify({"status": task.state})
    
@app.route("/results")
def results():
    filename = request.args.get("filename")
    if not filename:
        return "No file specified!", 400
    
    return render_template("results.html", filename=filename)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    prospective_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "prospective_students.csv")
    current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_students.csv")

    # Serve the file
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

    # Cleanup the files after download
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(prospective_file_path):
            os.remove(prospective_file_path)
        if os.path.exists(current_file_path):
            os.remove(current_file_path)
    except Exception as e:
        print(f"Error during cleanup: {e}")

    return response

if __name__ == "__main__":
    app.run(debug=True)
