import models.model as model
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def select_folder():
    folder_path = filedialog.askdirectory(title="Select a Folder")
    return folder_path

def select_image():
    image_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff")])
    return image_path

def output_test_result(img_rgb, img_with_box, hog_image, prediction):
    plt.figure(figsize = (18,6))

    # subplot 1
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Input Image")
    plt.axis('off')

    # subplot 2
    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box)
    plt.title(f"Predicted Expression {prediction} with detection")
    plt.axis('off')

    # subplot 3
    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG features")
    plt.axis('off')

    print(f"Predicted emotion is {prediction}")

    plt.show()


def show_ui():
    window = tk.Tk()
    window.title("Facial Emotion Recognition")
    window.geometry('500x600')

    #label = tk.Label(window, text="Upload your image here")
    #label.pack(pady=20)

    dataset1_path_var = tk.StringVar(value="No folder selected")
    dataset2_path_var = tk.StringVar(value="No folder selected")

    def select_dataset_folder1():
        folder = select_folder()
        if folder:
            dataset1_path_var.set(folder)

    dataset1_label = tk.Label(window, text="Select the first folder for the dataset")
    dataset1_label.pack(pady=5)

    dataset1_button = tk.Button(window, text="Select folder", command=select_dataset_folder1)
    dataset1_button.pack(pady=5)

    dataset2_label = tk.Label(window, text="Select the second folder for the dataset")
    dataset2_label.pack(pady=5)

    def select_dataset_folder2():
        folder = select_folder()
        if folder:
            dataset2_path_var.set(folder)

    dataset2_button = tk.Button(window, text="Select folder", command=select_dataset_folder2)
    dataset2_button.pack(pady=5)

    def submit_paths():
        print(f"First Dataset Path: {dataset1_path_var.get()}")
        print(f"Second Dataset Path: {dataset2_path_var.get()}")


    button = tk.Button(window, text="Check paths", command=submit_paths)
    button.pack(pady=20)

    accuracy_label = tk.Label(window, text="Accuracy: Not trained yet", fg="blue")
    accuracy_label.pack(pady=5)

    #confusion_matrix_label = tk.Label(window, text="Confusion Matrix: Not available", fg="blue", justify="left")
    #confusion_matrix_label.pack(pady=5)

    def train_model():
        dataset1_path = dataset1_path_var.get()
        dataset2_path = dataset2_path_var.get()
        #tree_classifier1, acc1, conf_matrix1 = model.train_model(dataset1_path)
        #tree_classifier2, acc2, conf_matrix2 = model.train_model(dataset2_path)

        tree_classifier, acc, conf_matrix = model.train_model([dataset1_path, dataset2_path])
        #Change signature of model.train_model
        print("Model has been trained")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Confusion Matrix:\n{conf_matrix}")

        global classifier
        classifier = tree_classifier
        global accuracy
        accuracy = acc
        global confusion_matrix
        confusion_matrix = conf_matrix

        accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
        #confusion_matrix_label.config(text=f"Confusion Matrix:\n{confusion_matrix}")

    train_button = tk.Button(window, text="Train model", command=train_model)
    train_button.pack(pady=20)

    def test_model_single():
        image_path = select_image()
        print(f"Image Path: {image_path}")
        img_rgb, img_with_box, hog_image, prediction = model.test_with_single_image(image_path, classifier)
        prediction_label.config(text=f"Predicted emotion: {prediction}")
        output_test_result(img_rgb, img_with_box, hog_image, prediction)


    test_button = tk.Button(window, text="Select path to test model with an image", command=test_model_single)
    test_button.pack(pady=20)

    prediction_label = tk.Label(window, text="Predicted Emotion: No prediction done", fg="blue")
    prediction_label.pack(pady=5)

    close_button = tk.Button(window, text="Close", command=window.destroy)
    close_button.pack(pady=30)

    window.mainloop()