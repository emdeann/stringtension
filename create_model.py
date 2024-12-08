import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage import io, color, transform
import joblib


# Load images and labels
def load_images_and_labels(data_dir):
    images = []
    labels = []
    for label, dirname in enumerate(os.listdir(data_dir)):
        if dirname.startswith('.'):  # ignore hidden files
            continue
        dirpath = os.path.join(data_dir, dirname)
        for filename in os.listdir(dirpath):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = io.imread(os.path.join(dirpath, filename))
                img = color.rgb2gray(img[...,0:3])
                img = transform.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)


# Load data from the directory
data_dir = 'standing_wave_images'
images, labels = load_images_and_labels(data_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Test different values of 'gamma' and 'C'
classifier = SVC()
grid = GridSearchCV(classifier, [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}])
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_prediction = best_model.predict(X_test)
best_accuracy = accuracy_score(y_prediction, y_test)

# Print the best accuracy score
print(f"Best Accuracy: {best_accuracy}")

# Save the model
joblib.dump(best_model, 'standing_wave_classifier.pkl', compress=3)
