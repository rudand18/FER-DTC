import cv2
import numpy as np
import os

from pyexpat import features
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from skimage.feature import hog

def extract_hog_features(images):
    hog_features = []
    for img in images:
        img = cv2.resize(img, (128,128))
        hog_feat = hog(img, orientations=8, pixels_per_cell=(16,16),
        cells_per_block=(4,4), block_norm='L2-Hys')
        hog_features.append(hog_feat)
    return np.array(hog_features)

def load_and_detect_faces(folder_path):
    face_cascade = cv2.CascadeClassifier('C:\\Users\\user\OneDrive - UCLan\ArtificialIntelligence\Assignment1\\fer\\resources\haarcascade_frontalface_default.xml')
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(30,30))
                    for(x, y, w, h) in faces:
                        face_region = img[y:y+h, x:x+w]
                        images.append(face_region)
                        labels.append(label)
    return images, labels


def train_model(dataset_paths):
    X_train_faces = []
    y_train = []
    X_test_faces = []
    y_test = []

    for dataset_path in dataset_paths:
        train_folder_path = os.path.join(dataset_path, 'train')

        if os.path.exists(train_folder_path):
            faces, labels = load_and_detect_faces(train_folder_path)
            X_train_faces.extend(faces)
            y_train.extend(labels)
        else:
            print(f"Warning: 'train' folder not found in {dataset_path}")

    X_train_features = extract_hog_features(X_train_faces)

    for dataset_path in dataset_paths:
        test_folder_path = os.path.join(dataset_path, 'test')

        if os.path.exists(test_folder_path):
            faces, labels = load_and_detect_faces(test_folder_path)
            X_test_faces.extend(faces)
            y_test.extend(labels)
        else:
            print(f"Warning: 'train' folder not found in {dataset_path}")

    X_test_features = extract_hog_features(X_test_faces)

    tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=10)
    tree_classifier.fit(X_train_features, y_train)

    y_pred = tree_classifier.predict(X_test_features)
    acc = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {acc*100:.2f}%")

    conf_matrix = confusion_matrix(y_test, y_pred)
    #print(f"Confusion Matrix:\n{conf_matrix}")

    return tree_classifier, acc, conf_matrix

def extract_hog_features_single(img):
    img = cv2.resize(img, (128,128))

    hog_features, hog_image = hog(img, orientations=8, pixels_per_cell=(16,16),cells_per_block=(4,4), block_norm='L2-Hys', visualize=True)
    return hog_features, hog_image

def predict_emotion_hog(image_path, classifier):

    face_cascade = cv2.CascadeClassifier('C:\\Users\\user\OneDrive - UCLan\ArtificialIntelligence\Assignment1\\fer\\resources\haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image {image_path} cannot be opened")

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(30,30))

    if len(faces) == 0:
        raise ValueError(f"No faces detected")

    (x,y,w,h) = faces[0]
    face_region = img[y:y+h, x:x+w]

    features, hog_image = extract_hog_features_single(face_region)

    features = features.reshape(1, -1)

    prediction = classifier.predict(features)[0]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_with_box, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return img_rgb, img_with_box, hog_image, prediction

def test_with_single_image(image_path, classifier):
    img_rgb, img_with_box, hog_image, prediction = predict_emotion_hog(image_path, classifier)
    return img_rgb, img_with_box, hog_image, prediction
