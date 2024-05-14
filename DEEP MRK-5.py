from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten ,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.callbacks import EarlyStopping
# import scipy

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel('ERROR')

total_samples = 1584 + 1132 + 869 + 585
class1_samples = 1584
class2_samples = 1132
class3_samples = 869
class4_samples = 585

# Calculate weights
weight1 = total_samples / class1_samples
weight2 = total_samples / class2_samples
weight3 = total_samples / class3_samples
weight4 = total_samples / class4_samples

# Normalize weights
total_weight = weight1 + weight2 + weight3 + weight4
weight1 /= total_weight
weight2 /= total_weight
weight3 /= total_weight
weight4 /= total_weight

# Set class weights
class_weights = {0: weight1, 1: weight2, 2: weight3, 3: weight4}

# Define image dimensions
image_height, image_width = 224, 224
num_classes = 4
batch_size = 32
epochs = 50

# Define paths to your training and validation data
train_data_dir = 'D:/python/deep-learning-project/deep learning Ulcerative Colitis dataset/train_and_validation_sets/train_and_validation_sets'
val_data_dir = 'D:/python/deep-learning-project/deep learning Ulcerative Colitis dataset/test_set/test_set'


def lenet():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(image_height, image_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)  # Shuffle the data

    val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)  # Don't shuffle validation data
    model = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define a learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)


    # Define a checkpoint to save the best model during training
    checkpoint = ModelCheckpoint('best_model_lenet.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=[checkpoint, lr_scheduler, early_stopping],
                        class_weight=class_weights)

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()


def resnet50():
    # Use ImageDataGenerator for preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Load ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add dropout to reduce overfitting
    x = Dropout(0.5)(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)

    # Add dropout to reduce overfitting
    x = Dropout(0.5)(x)

    # Add a classification layer for num_classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # Model to be trained
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the original ResNet50 model up to a certain layer
    for layer in base_model.layers[:140]:
        layer.trainable = False
    for layer in base_model.layers[140:]:
        layer.trainable = True

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('best_model_resnet.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='ResNet50 Training Loss')
    plt.plot(history.history['val_loss'], label='ResNet50 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
def alexnet():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(image_height, image_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)  # Shuffle the data

    val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                    target_size=(image_height, image_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)  # Don't shuffle validation data

    # Define AlexNet model
    model = Sequential([
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define a learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)

    # Define a checkpoint to save the best model during training
    checkpoint = ModelCheckpoint('best_model_alexnet.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        callbacks=[checkpoint, lr_scheduler,early_stopping],
                        class_weight=class_weights)

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    
    
def inception_v3():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
    

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('inception_v3.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='InceptionV3 Training Loss')
    plt.plot(history.history['val_loss'], label='InceptionV3 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def xception():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('xception.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Xception Training Loss')
    plt.plot(history.history['val_loss'], label='Xception Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def mobilenet():
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('mobilenet.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='MobileNet Training Loss')
    plt.plot(history.history['val_loss'], label='MobileNet Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def mobilenet_v2():
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('mobilenet_v2.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='MobileNetV2 Training Loss')
    plt.plot(history.history['val_loss'], label='MobileNetV2 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def densenet121():
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('densenet121.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='DenseNet121 Training Loss')
    plt.plot(history.history['val_loss'], label='DenseNet121 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    
def densenet169():
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('densenet169.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='DenseNet169 Training Loss')
    plt.plot(history.history['val_loss'], label='DenseNet169 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def densenet201():
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Generate batches of augmented data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Adding custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('densenet201.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='DenseNet201 Training Loss')
    plt.plot(history.history['val_loss'], label='DenseNet201 Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def vgg16():
    mixed_precision.set_global_policy('mixed_float16')
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator()
    
    # Generate data batches
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('best_model_vgg16.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    
def vgg19():
    mixed_precision.set_global_policy('mixed_float16')

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator()
    
    # Generate data batches
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load pre-trained VGG19 model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max', verbose=1)
    
    checkpoint = ModelCheckpoint('best_model_vgg19.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, lr_scheduler,early_stopping],
        class_weight=class_weights,
        epochs=epochs
    )
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


vgg19()
vgg16()
densenet201()
densenet169()
densenet121()
mobilenet_v2()
mobilenet()
xception()
inception_v3()
resnet50()
lenet()
alexnet()

#Developed a Deep Learning model using vgg19, vgg16, densenet201, densenet169, densenet121, mobilenet_v2, mobilenet, xception,inception_v3, resnet50, lenet, alexnet



