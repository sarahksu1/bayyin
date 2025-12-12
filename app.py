from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# =========================
# تحميل المودل
# =========================
MODEL_PATH = "model/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# إعداد مجلد رفع الصور
# =========================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# تجهيز الصورة
# =========================
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# الصفحة الرئيسية
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# =========================
# صفحة النتائج
# =========================
@app.route("/dashboard", methods=["POST"])
def dashboard_result():

    if "image" not in request.files:
        return "❌ لم يتم رفع صورة"

    file = request.files["image"]
    if file.filename == "":
        return "❌ اسم الملف فارغ"

    # حفظ الصورة
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # تشغيل المودل
    img_array = prepare_image(img_path)
    prediction = float(model.predict(img_array)[0][0])

    # =========================
    # حساب نسبة الاشتباه (0–100)
    # =========================
    score = int((1 - prediction) * 100)
    score = max(0, min(score, 100))

    # =========================
    # تحديد الحكم
    # =========================
    if prediction >= 0.6:
        status = "مزور ❗"
        fraudType = "AI"
        ring_color = "red"
    else:
        status = "سليم ✔"
        fraudType = None
        ring_color = "green"

    # =========================
    # عرض النتائج
    # =========================
    return render_template(
        "dashboard.html",
        score=score,
        status=status,
        fraudType=fraudType,
        ring_color=ring_color,
        summary="تم تحليل الصورة باستخدام نموذج ذكاء اصطناعي مدرّب",
        confidence=f"{score}%"
    )

# =========================
# تشغيل التطبيق
# =========================
if __name__ == "__main__":
    app.run(debug=True)
