import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# ==============================
# MODEL
# ==============================

MODEL_PATH = "interior_scene_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ==============================
# CLASS NAMES
# ==============================

class_names = [
    'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar',
    'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet',
    'casino', 'children_room', 'church_inside', 'classroom',
    'cloister', 'closet', 'clothingstore', 'computerroom',
    'concert_hall', 'corridor', 'deli', 'dentaloffice',
    'dining_room', 'elevator', 'fastfood_restaurant', 'florist',
    'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym',
    'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway',
    'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet',
    'laundromat', 'library', 'livingroom', 'lobby', 'locker_room',
    'mall', 'meeting_room', 'movietheater', 'museum', 'nursery',
    'office', 'operating_room', 'pantry', 'poolinside',
    'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop',
    'stairscase', 'studiomusic', 'subway', 'toystore',
    'trainstation', 'tv_studio', 'videostore', 'waitingroom',
    'warehouse', 'winecellar'
]


# ==============================
# GET IMAGE
# ==============================

image_path = input("Enter the image path: ")

img = load_img(image_path, target_size=(224, 224))

img_array = img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    img_array
)


# ==============================
# PREDICT
# ==============================

predictions = model.predict(img_array, verbose=0)
# Get top 5 predictions
top_indices = np.argsort(predictions[0])[-5:][::-1]

print("\n==============================")
print("🏠 AI INTERIOR ANALYSIS")
print("==============================")

print("Top 5 Predictions:\n")

for index in top_indices:
    predicted_class = class_names[index]
    confidence = predictions[0][index] * 100

    print(f"{predicted_class:<25} {confidence:.2f}%")

print("==============================")


# ==============================
# RESULT
# ==============================

print("\n==============================")
print("🏠 AI INTERIOR ANALYSIS")
print("==============================")

print(f"Predicted Scene: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print("==============================")