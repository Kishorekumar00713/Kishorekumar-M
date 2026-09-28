import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# DATASET PATH
# ==============================

DATASET_PATH = r"C:\Users\archa\Downloads\archive (1)\indoorCVPR_09\Images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


# ==============================
# LOAD DATASET
# ==============================

print("Loading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ==============================
# GET CLASS NAMES
# ==============================

class_names = train_dataset.class_names

print("\nNumber of classes:", len(class_names))
print("Classes:")
print(class_names)


# ==============================
# PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.apply(
    tf.data.experimental.ignore_errors()
)

validation_dataset = validation_dataset.apply(
    tf.data.experimental.ignore_errors()
)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ==============================
# PRE-TRAINED MODEL
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==============================
# BUILD OUR MODEL
# ==============================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)


# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# TRAIN
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)


# ==============================
# SAVE MODEL
# ==============================

model.save("interior_scene_model.keras")

print("\n================================")
print("MODEL TRAINING COMPLETE! ✅")
print("Model saved as:")
print("interior_scene_model.keras")
print("================================")
# ==============================
# FINAL MODEL EVALUATION
# ==============================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(validation_dataset)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")