import sys
sys.stdout.reconfigure(encoding='utf-8') #again, using this for encoding symbols that may otherwise not be visible during output
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt

# Loading the pre-trained model we saved earlier
model = tf.keras.models.load_model('malaria_cnn_model.h5')  

# The function that takes an image file path and prepares it for our model by resizing it and normalizing the pixel values.
def preprocess_image(img_path, target_size=(150, 150)):
    img = Image.open(img_path)  # Opening the image file.
    img = img.resize(target_size)  # Resize the image to match the input size expected by our model.
    img_array = image.img_to_array(img)  # Convert the image to a numpy array format, which is easier to work with.
    img_array = np.expand_dims(img_array, axis=0)  # Add a new axis so that it's shaped correctly for prediction (1, 150, 150, 3).
    img_array /= 255.0  # Scale the pixel values to be between 0 and 1, as our model was trained with this normalization.
    return img_array

# This function takes in the preprocessed image and runs it through our model to get a prediction.
def predict_image(model, img_array):
    prediction = model.predict(img_array)  # Using the model made in the other program to predict whether the cell is infected or not.
    return prediction

# The main function that handles everything, it lets us select an image, processes it, and shows the prediction result.
def main():
    # Setting up tkinter but keeping it hidden for now, so we can use the file dialog without showing an empty GUI window.
    root = Tk()
    root.withdraw()  # Not showing the root window, we're only using it for the file dialog.
    root.call('wm', 'attributes', '.', '-topmost', True)  # Making sure the file dialog pops up on top of other windows.

    # This opens a file dialog to pick an image file. We can choose JPEG or PNG files
    img_path = filedialog.askopenfilename(title="Select an Image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    # Checking here if the user selected a file
    if img_path:
        # Preprocessing the image so it's ready to be fed into the model
        img_array = preprocess_image(img_path)

        # Run the image through the model to get a prediction
        prediction = predict_image(model, img_array)

        # Displaying the selected image using matplotlib (just for utility, can be removed later)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')  # Turning off the axis on the graph for a cleaner look.
        plt.show()  # Show the image in a new window.

        # Print the result based on the prediction which is either "Infected" or "Uninfected" along with the confidence score.
        if prediction[0][0] > 0.5:
            print(f"The cell is predicted to be UNINFECTED with a confidence of {prediction[0][0]*100:.2f}%.")
        else:
            print(f"The cell is predicted to be INFECTED with a confidence of {(1 - prediction[0][0])*100:.2f}%.")
    else:
        print("No image was selected.")  # If no file was selected, just let the user know and stop prgram

# This is the starting point of the script and it runs the main function
if __name__ == "__main__":
    main()
