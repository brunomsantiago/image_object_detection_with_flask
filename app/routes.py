from flask import (render_template,
                   flash,
                   send_from_directory,
                   redirect, request)
from app import app, detector
from PIL import Image
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/index',  methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            print(' - no file')
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert('RGB')
            notes = detector.detect_and_get_notes(image)
            return render_template('index.html',
                                   filename=filename,
                                   notes=notes)

    return render_template('index.html')


@app.route('/quick_cache/<filename>')
def download_file(filename):
    print(filename)
    return send_from_directory('uploaded_images', filename, as_attachment=True)
