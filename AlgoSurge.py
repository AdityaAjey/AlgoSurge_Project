import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import cv2


# Generate synthetic images with simple shapes
def generate_images(num_samples, image_size):
    images = []
    masks = []
    for i in range(num_samples):
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        
        # Randomly select shape (circle or square)
        shape = np.random.choice(['circle', 'square'])
        
        # Randomly generate shape parameters
        if shape == 'circle':
            center = np.random.randint(20, image_size-20, size=2)
            radius = np.random.randint(10, 20)
            cv2.circle(image, tuple(center), radius, (255, 255, 255), -1)
            cv2.circle(mask, tuple(center), radius, 255, -1)
        elif shape == 'square':
            top_left = np.random.randint(10, image_size-30, size=2)
            bottom_right = top_left + np.random.randint(20, 30, size=2)
            cv2.rectangle(image, tuple(top_left), tuple(bottom_right), (255, 255, 255), -1)
            cv2.rectangle(mask, tuple(top_left), tuple(bottom_right), 255, -1)
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Define the CNN model architecture
def create_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv3 = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(conv2)
    model = keras.Model(inputs=inputs, outputs=conv3)
    return model

# Generate synthetic dataset
num_samples = 1000
image_size = 64
images, masks = generate_images(num_samples, image_size)

# Normalize images to [0, 1]
images = images.astype('float32') / 255.0

# Split dataset into training and validation sets
train_split = int(0.8 * num_samples)
x_train, x_val = images[:train_split], images[train_split:]
y_train, y_val = masks[:train_split], masks[train_split:]

# Create the model
input_shape = (image_size, image_size, 3)
model = create_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Report key training metrics
print("Training Loss:", history.history['loss'][-1])
print("Validation Loss:", history.history['val_loss'][-1])

# Perform segmentations using the model on a few images from the validation set
num_samples_to_segment = 5
sample_indices = np.random.choice(len(x_val), num_samples_to_segment, replace=False)
for i, idx in enumerate(sample_indices):
    sample_image = x_val[idx]
    sample_mask = y_val[idx]
    predicted_mask = model.predict(sample_image[np.newaxis, ...])[0]

    # Plot the original image, ground truth mask, and predicted mask
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image)
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(sample_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
