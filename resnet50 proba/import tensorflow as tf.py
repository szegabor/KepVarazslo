import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Előre betanított ResNet50 modell betöltése, ImageNet súlyokkal
base_model = ResNet50(weights='imagenet', include_top=False)

# Új rétegek hozzáadása a modellhez
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Két osztály: Skittles és egyéb

# A teljes modell összeállítása
model = Model(inputs=base_model.input, outputs=predictions)

# Az alapmodell rétegeinek befagyasztása
for layer in base_model.layers:
    layer.trainable = False

# A modell kompilálása
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Adat augmentáció: a meglévő adatokból további variációk készítése
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,      # Képek forgatása
    brightness_range=[0.8, 1.2],  # Fényerő változtatás
    width_shift_range=0.2,  # Kép vízszintes eltolása
    height_shift_range=0.2, # Kép függőleges eltolása
    horizontal_flip=True    # Képek vízszintes tükrözése
)

# Betöltjük az adatokat a dataset mappából
train_generator = train_datagen.flow_from_directory(
    'dataset/',  # A mappa elérési útvonala
    target_size=(224, 224),  # Kép mérete
    batch_size=32,
    class_mode='categorical'
)

# A modell betanítása
model.fit(train_generator, epochs=20)

# Modell elmentése, hogy később ne kelljen újra betanítani
model.save('skittles_augmented_model.h5')
