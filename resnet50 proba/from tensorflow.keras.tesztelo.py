import tensorflow as tf
from tensorflow.keras.models import load_model

# Modell betöltése
model = load_model('skittles_augmented_model.h5')

# Képes predikció végrehajtása a betöltött modellel
def predict_image_class(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image = tf.expand_dims(image, axis=0)  # Bemeneti méret módosítása (batch dimension hozzáadása)

    # Modell predikció
    prediction = model.predict(image)
    
    # Osztály címkék
    class_labels = {0: "other", 1: "skittles"}
    predicted_class = class_labels[prediction.argmax()]
    
    print(f"The predicted class for the image is: {predicted_class}")

# Tesztel egy képpel
predict_image_class(r'c:\Users\varda\Documents\szakdolgozat\opencv app\proba kod\dataset\ga.jpg')
