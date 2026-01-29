import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape=(224, 224, 3), num_classes=2):
    # Load base model
    vgg = VGG19(include_top=False, input_shape=input_shape, weights='imagenet')
    
    # Freeze the pre-trained layers
    for layer in vgg.layers:
        layer.trainable = False
        
    # Add custom head
    x = Flatten()(vgg.output)
    prediction = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=vgg.input, outputs=prediction)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def train_model(train_path, test_path, model_save_path='model_new.h5'):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Training or test paths do not exist:\n  {train_path}\n  {test_path}")
        print("Please ensure the dataset is downloaded and extracted.")
        return

    # Data Augmentation & Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model = build_model()
    model.summary()

    print("Starting training...")
    model.fit(
        train_set,
        validation_data=test_set,
        epochs=10,
        verbose=1
    )

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def download_dataset(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    urls = {
        "train": "https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip",
        "validation": "https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip"
    }
    
    for category, url in urls.items():
        dst_folder = os.path.join(target_dir, category)
        if not os.path.exists(dst_folder):
            zip_path = os.path.join(target_dir, f"{category}.zip")
            print(f"Downloading {category} dataset...")
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dst_folder)
            os.remove(zip_path)
            print(f"Extracted {category} dataset to {dst_folder}")

if __name__ == "__main__":
    # Update these paths if your dataset location differs
    DATA_ROOT = os.path.join(os.getcwd(), 'horse-or-human')
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    TEST_DIR = os.path.join(DATA_ROOT, 'validation')
    
    # Download dataset if missing
    download_dataset(DATA_ROOT)
    
    train_model(TRAIN_DIR, TEST_DIR)
