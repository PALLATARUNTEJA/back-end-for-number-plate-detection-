##
# @app.route('/crop', methods=['POST'])
# def crop_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['CROPPED_FOLDER'], filename)
#     file.save(file_path)

#     # Perform OCR on the cropped image
#     result = ocr.ocr(file_path)
#     text = ""
#     if result and len(result) > 0:
#         for line in result:
#             if line and len(line) > 0:
#                 for word in line:
#                     if word and len(word) > 1:
#                         text += word[1][0] + " "

#     # Filter out unwanted characters and extract the most likely number plate text
#     filtered_text = filter_number_plate_text(text)
#     return jsonify({
#         "result": filtered_text if filtered_text else "No valid number plate text detected",
#         "image": filename
#     })
