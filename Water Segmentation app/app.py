import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
import tensorflow as tf
import tifffile as tiff
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# إعداد Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# تحميل الموديل مع IOU metric
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

model = tf.keras.models.load_model("models/unet_water.h5", custom_objects={'iou_metric': iou_metric})

# دالة التنبؤ والرسم
def predict_and_plot(filepath):
    # قراءة الصورة .tif
    img = tiff.imread(filepath)
    if img.shape[0] == 12:
        img = np.moveaxis(img, 0, -1)

    # Resizing + Normalization
    resized_img = tf.image.resize(img, (128, 128)).numpy()
    norm_img = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min() + 1e-7)
    norm_img = np.expand_dims(norm_img, axis=0)

    # التنبؤ بالماسك
    pred_mask = model.predict(norm_img)[0, :, :, 0]

    # حفظ نسخة RGB من الصورة الأصلية كـ PNG
    rgb = img[:, :, :3]
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

    base = os.path.splitext(os.path.basename(filepath))[0]
    out_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base}_input.png')
    out_mask_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base}_mask.png')

    plt.imsave(out_img_path, rgb_uint8)
    plt.imsave(out_mask_path, pred_mask, cmap='gray')

    return os.path.basename(out_img_path), os.path.basename(out_mask_path)


# الراوت الأساسي
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_name, mask_name = predict_and_plot(filepath)

            return render_template('result.html', image_file=image_name, mask_file=mask_name)
    return render_template('index.html')


# استدعاء الصور
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)
