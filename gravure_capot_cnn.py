import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Définition des paramètres
image_height = 791
image_width = 1036

num_channels = 3 # Nombre de canaux pour les images en couleur (RGB)
num_classes = 2
batch_size = 16  # Réduction de la taille du batch
epochs = 10

# Chemins vers les répertoires contenant les données d'entraînement et de test
train_dir =r"C:\Users\INFOKOM\Desktop\Dataset\gravure capot\train_gravure capot"
test_dir = r"C:\Users\INFOKOM\Desktop\Dataset\gravure capot\test_gravure capot"

# Création des générateurs de données pour l'entraînement et le test
train_datagen = ImageDataGenerator(rescale=1./255)  
test_datagen = ImageDataGenerator(rescale=1./255)  



# Confirmation de la création des générateurs de données
print("Générateurs de données créés avec succès.")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Confirmation de la création des générateurs de données d'entraînement et de test
print("Générateurs de données d'entraînement et de test créés avec succès.")

# Création du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Confirmation de la création du modèle CNN
print("Modèle CNN créé avec succès.")

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Confirmation de la compilation du modèle
print("Modèle compilé avec succès.")

# Entraînement du modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Confirmation de l'entraînement du modèle
print("Modèle entraîné avec succès.")

# Sauvegarde du modèle au format .h5
model.save("gravurecapot_cnn.h5")
print("Modèle sauvegardé au format .h5 avec succès.")

# Évaluation du modèle
test_loss, test_acc = model.evaluate(
    test_generator,
    steps=test_generator.samples // batch_size
)

print('Test accuracy:', test_acc)
print("Modèle évalué avec succès.")

# Obtenir les historiques d'entraînement
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot des courbes d'apprentissage pour la précision
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot des courbes d'apprentissage pour la perte
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
