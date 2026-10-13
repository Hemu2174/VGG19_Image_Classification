# 🖼️ Image Classification Using VGG-19 and Streamlit

A deep learning-based image classification web application built using VGG-19, PyTorch, and Streamlit. The application uses a pretrained VGG-19 convolutional neural network to identify objects in uploaded images and display the top-5 predicted categories with their confidence scores.

## 📌 Project Overview

Image classification is a computer vision task that identifies the objects or categories present in an image. This project demonstrates how a pretrained deep learning model can be integrated into an interactive web application.

Users can upload an image in JPG, JPEG, or PNG format. The application preprocesses the image, passes it through the pretrained VGG-19 model, and displays the five most likely ImageNet categories.

- **Project type:** Academic Mini Project
- **Domain:** Artificial Intelligence, Deep Learning, Computer Vision
- **Model:** VGG-19
- **Interface:** Streamlit

## ✨ Features

- 🖼️ Upload images in JPG, JPEG, and PNG formats.
- 🧠 Classify images using pretrained VGG-19 weights.
- 🏷️ Display the top-1 predicted category.
- 📊 Show confidence scores for the top-5 predictions.
- 📋 Present predictions in a readable table.
- 📈 Visualize prediction confidence using a bar chart.
- 🔍 Display image metadata, including dimensions, color mode, and file size.
- ⚡ Cache the model to avoid unnecessary reloads during app interactions.
- 🎨 Provide an interactive interface with a sidebar and custom styling.
- 🛡️ Validate uploaded images before processing.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework and inference |
| Torchvision | VGG-19 model and image transformations |
| Streamlit | Interactive web application |
| Pillow (PIL) | Image loading and processing |
| Pandas | Prediction tables and result handling |
| Matplotlib / Streamlit charts | Prediction visualization, as implemented |

## 🧠 Model Architecture: VGG-19

VGG-19 is a convolutional neural network introduced by the Visual Geometry Group at the University of Oxford. It is commonly used for image recognition and classification.

The pretrained VGG-19 model used in this project was trained on ImageNet, which contains 1,000 classification categories.

### Architecture Overview

- **Input layer:** Accepts an RGB image resized to 224 × 224 pixels.
- **Convolutional layers:** Extract visual features such as edges, textures, patterns, and shapes.
- **ReLU activation:** Introduces non-linearity.
- **Max-pooling layers:** Reduce spatial dimensions while retaining important features.
- **Fully connected layers:** Use the extracted features to produce classification scores.
- **Output layer:** Produces scores for 1,000 ImageNet categories.

### Why VGG-19?

- It is a well-established CNN architecture.
- Pretrained weights allow classification without training a model from scratch.
- Its architecture is useful for learning convolutional neural networks.
- It provides a practical introduction to transfer learning and model inference.

**Note:** This project uses pretrained ImageNet weights. It does not train VGG-19 on a custom dataset.

## 🔄 Application Workflow

1. Upload an image through the Streamlit interface.
2. Validate the uploaded image.
3. Preprocess the image to match the model's expected input format.
4. Pass the processed image through the pretrained VGG-19 model.
5. Generate classification scores for ImageNet categories.
6. Identify the top-5 predicted categories.
7. Display predictions, confidence scores, a table, and a chart.

## 🧩 Image Preprocessing

The uploaded image must be converted into the format expected by the pretrained model.

The preprocessing pipeline consists of the following steps:

1. Load the uploaded image using Pillow.
2. Convert the image to RGB format.
3. Resize it to 224 × 224 pixels.
4. Convert the image into a PyTorch tensor.
5. Normalize the tensor using the ImageNet mean and standard deviation.
6. Pass the processed tensor to VGG-19 for inference.

Preprocessing is important because the pretrained model expects images in a particular format and numerical range.

## 📂 Project Structure

```text
VGG19_Image_Classification/
│
├── app.py              # Streamlit application and prediction logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Files excluded from Git
└── venv/               # Local virtual environment (not committed)
```

The VGG-19 model weights are downloaded and cached by Torchvision when required. They do not need to be included in the GitHub repository.

## 🚀 How to Use the Application

1. Open the deployed Streamlit application.
2. Select an image using the file uploader.
3. Wait for the image to be processed by VGG-19.
4. View the uploaded image and the top predicted category.
5. Review the confidence scores of the top-5 predictions.
6. Examine the prediction table and confidence chart.

The first run may take longer because Torchvision needs to download the pretrained model weights. Subsequent runs can reuse the locally cached weights.

**Live Application:** [VGG19 Image Classification](https://vgg19imageclassification.streamlit.app/)

## 📊 Understanding the Results

The application displays the predicted ImageNet categories and their associated confidence scores.

- **Top-1 prediction:** The category with the highest predicted probability.
- **Top-5 predictions:** The five categories with the highest predicted probabilities.
- **Confidence score:** The model's estimated probability for a particular category.

For example, an image of a fruit might produce a fruit-related category among the top predictions.

The exact categories and scores depend on the uploaded image.

**Important:** A confidence score is not a guarantee that a prediction is correct. Model predictions can be affected by image quality, lighting, object visibility, and similarity between categories.

## ⚠️ Limitations

- The model recognizes categories from its pretrained ImageNet label set.
- It cannot reliably identify arbitrary custom categories that are not represented in its training classes.
- The project does not include custom model training or fine-tuning.
- Prediction results depend on image quality and preprocessing.
- Inference on a CPU may be slower than on a compatible GPU.
- The application provides model predictions, not guaranteed ground-truth labels.

## 🔮 Future Enhancements

- Add a custom dataset and train or fine-tune the model.
- Compare VGG-19 with ResNet, MobileNet, and EfficientNet.
- Add Grad-CAM visualizations to highlight image regions influencing predictions.
- Support batch image classification.
- Export prediction results as CSV.
- Add additional model evaluation metrics using a labeled test dataset.
- Improve inference performance and add optional GPU support.

## 🎓 Viva Questions and Answers

### 1. What is image classification?

Image classification is a computer vision task that assigns a category or label to an image based on its visual content.

### 2. What is VGG-19?

VGG-19 is a convolutional neural network architecture developed by the Visual Geometry Group at the University of Oxford. Its name refers to the 19 weight-bearing layers.

### 3. What is a CNN?

A Convolutional Neural Network (CNN) is a deep learning model designed to learn spatial features from images using convolutional filters and related operations.

### 4. Why is VGG-19 used in this project?

VGG-19 provides a pretrained image classification model that can recognize ImageNet categories without requiring the project to train a network from scratch.

### 5. What is transfer learning?

Transfer learning reuses knowledge learned by a model on one task or dataset for another task. Using pretrained weights is a common transfer learning approach.

### 6. What is ImageNet?

ImageNet is a large image dataset commonly used to train and evaluate image recognition models. The standard pretrained VGG-19 classifier predicts among 1,000 ImageNet categories.

### 7. Why is an image resized to 224 × 224?

The standard pretrained VGG-19 image classification pipeline expects a 224 × 224 input image, so resizing provides a consistent input size.

### 8. Why convert an image to RGB?

The model expects three color channels. Converting to RGB ensures a consistent color representation for images that may otherwise be grayscale or use another color mode.

### 9. What is normalization?

Normalization adjusts pixel values using predefined statistics, such as the ImageNet channel means and standard deviations, to match the preprocessing expected by the pretrained model.

### 10. What is Streamlit?

Streamlit is a Python framework for building interactive web applications, particularly for data science and machine learning projects.

### 11. What is PyTorch?

PyTorch is a deep learning framework used to create, train, and run neural networks. In this project, it performs model inference.

### 12. What is the top-5 prediction?

The top-5 prediction shows the five categories with the highest model output probabilities for an input image.

### 13. Does this project train the VGG-19 model?

No. The project uses pretrained ImageNet weights for inference and does not train the model on a custom dataset.

### 14. What is the purpose of softmax?

Softmax converts a vector of classification logits into normalized probability values. In this project, these values can be used to rank ImageNet categories.

### 15. What happens during the first run?

The pretrained model weights may be downloaded and cached locally. Once the weights are available, later runs can reuse the cached model.

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision VGG-19 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pillow Documentation](https://pillow.readthedocs.io/)

## 👨‍💻 Authors

### 1. Kattamuri Hemanth

- GitHub: [Hemu2174](https://github.com/Hemu2174)

### 2. CHOKKAPU DHANA LAKSHMI

- GitHub: [dhana3108](https://github.com/dhana3108)

---

⭐ If you find this project useful for learning computer vision and deep learning, consider starring the repository.# 🖼️ Image Classification Using VGG-19 and Streamlit

A deep learning-based image classification web application built using VGG-19, PyTorch, and Streamlit. The application uses a pretrained VGG-19 convolutional neural network to identify objects in uploaded images and display the top-5 predicted categories with their confidence scores.

## 📌 Project Overview

Image classification is a computer vision task that identifies the objects or categories present in an image. This project demonstrates how a pretrained deep learning model can be integrated into an interactive web application.

Users can upload an image in JPG, JPEG, or PNG format. The application preprocesses the image, passes it through the pretrained VGG-19 model, and displays the five most likely ImageNet categories.

- **Project type:** Academic Mini Project
- **Domain:** Artificial Intelligence, Deep Learning, Computer Vision
- **Model:** VGG-19
- **Interface:** Streamlit

## ✨ Features

- 🖼️ Upload images in JPG, JPEG, and PNG formats.
- 🧠 Classify images using pretrained VGG-19 weights.
- 🏷️ Display the top-1 predicted category.
- 📊 Show confidence scores for the top-5 predictions.
- 📋 Present predictions in a readable table.
- 📈 Visualize prediction confidence using a bar chart.
- 🔍 Display image metadata, including dimensions, color mode, and file size.
- ⚡ Cache the model to avoid unnecessary reloads during app interactions.
- 🎨 Provide an interactive interface with a sidebar and custom styling.
- 🛡️ Validate uploaded images before processing.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework and inference |
| Torchvision | VGG-19 model and image transformations |
| Streamlit | Interactive web application |
| Pillow (PIL) | Image loading and processing |
| Pandas | Prediction tables and result handling |
| Matplotlib / Streamlit charts | Prediction visualization, as implemented |

## 🧠 Model Architecture: VGG-19

VGG-19 is a convolutional neural network introduced by the Visual Geometry Group at the University of Oxford. It is commonly used for image recognition and classification.

The pretrained VGG-19 model used in this project was trained on ImageNet, which contains 1,000 classification categories.

### Architecture Overview

- **Input layer:** Accepts an RGB image resized to 224 × 224 pixels.
- **Convolutional layers:** Extract visual features such as edges, textures, patterns, and shapes.
- **ReLU activation:** Introduces non-linearity.
- **Max-pooling layers:** Reduce spatial dimensions while retaining important features.
- **Fully connected layers:** Use the extracted features to produce classification scores.
- **Output layer:** Produces scores for 1,000 ImageNet categories.

### Why VGG-19?

- It is a well-established CNN architecture.
- Pretrained weights allow classification without training a model from scratch.
- Its architecture is useful for learning convolutional neural networks.
- It provides a practical introduction to transfer learning and model inference.

**Note:** This project uses pretrained ImageNet weights. It does not train VGG-19 on a custom dataset.

## 🔄 Application Workflow

1. Upload an image through the Streamlit interface.
2. Validate the uploaded image.
3. Preprocess the image to match the model's expected input format.
4. Pass the processed image through the pretrained VGG-19 model.
5. Generate classification scores for ImageNet categories.
6. Identify the top-5 predicted categories.
7. Display predictions, confidence scores, a table, and a chart.

## 🧩 Image Preprocessing

The uploaded image must be converted into the format expected by the pretrained model.

The preprocessing pipeline consists of the following steps:

1. Load the uploaded image using Pillow.
2. Convert the image to RGB format.
3. Resize it to 224 × 224 pixels.
4. Convert the image into a PyTorch tensor.
5. Normalize the tensor using the ImageNet mean and standard deviation.
6. Pass the processed tensor to VGG-19 for inference.

Preprocessing is important because the pretrained model expects images in a particular format and numerical range.

## 📂 Project Structure

```text
VGG19_Image_Classification/
│
├── app.py              # Streamlit application and prediction logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Files excluded from Git
└── venv/               # Local virtual environment (not committed)
```

The VGG-19 model weights are downloaded and cached by Torchvision when required. They do not need to be included in the GitHub repository.

## 🚀 How to Use the Application

1. Open the deployed Streamlit application.
2. Select an image using the file uploader.
3. Wait for the image to be processed by VGG-19.
4. View the uploaded image and the top predicted category.
5. Review the confidence scores of the top-5 predictions.
6. Examine the prediction table and confidence chart.

The first run may take longer because Torchvision needs to download the pretrained model weights. Subsequent runs can reuse the locally cached weights.

**Live Application:** [VGG19 Image Classification](https://vgg19imageclassification.streamlit.app/)

## 📊 Understanding the Results

The application displays the predicted ImageNet categories and their associated confidence scores.

- **Top-1 prediction:** The category with the highest predicted probability.
- **Top-5 predictions:** The five categories with the highest predicted probabilities.
- **Confidence score:** The model's estimated probability for a particular category.

For example, an image of a fruit might produce a fruit-related category among the top predictions.

The exact categories and scores depend on the uploaded image.

**Important:** A confidence score is not a guarantee that a prediction is correct. Model predictions can be affected by image quality, lighting, object visibility, and similarity between categories.

## ⚠️ Limitations

- The model recognizes categories from its pretrained ImageNet label set.
- It cannot reliably identify arbitrary custom categories that are not represented in its training classes.
- The project does not include custom model training or fine-tuning.
- Prediction results depend on image quality and preprocessing.
- Inference on a CPU may be slower than on a compatible GPU.
- The application provides model predictions, not guaranteed ground-truth labels.

## 🔮 Future Enhancements

- Add a custom dataset and train or fine-tune the model.
- Compare VGG-19 with ResNet, MobileNet, and EfficientNet.
- Add Grad-CAM visualizations to highlight image regions influencing predictions.
- Support batch image classification.
- Export prediction results as CSV.
- Add additional model evaluation metrics using a labeled test dataset.
- Improve inference performance and add optional GPU support.

## 🎓 Viva Questions and Answers

### 1. What is image classification?

Image classification is a computer vision task that assigns a category or label to an image based on its visual content.

### 2. What is VGG-19?

VGG-19 is a convolutional neural network architecture developed by the Visual Geometry Group at the University of Oxford. Its name refers to the 19 weight-bearing layers.

### 3. What is a CNN?

A Convolutional Neural Network (CNN) is a deep learning model designed to learn spatial features from images using convolutional filters and related operations.

### 4. Why is VGG-19 used in this project?

VGG-19 provides a pretrained image classification model that can recognize ImageNet categories without requiring the project to train a network from scratch.

### 5. What is transfer learning?

Transfer learning reuses knowledge learned by a model on one task or dataset for another task. Using pretrained weights is a common transfer learning approach.

### 6. What is ImageNet?

ImageNet is a large image dataset commonly used to train and evaluate image recognition models. The standard pretrained VGG-19 classifier predicts among 1,000 ImageNet categories.

### 7. Why is an image resized to 224 × 224?

The standard pretrained VGG-19 image classification pipeline expects a 224 × 224 input image, so resizing provides a consistent input size.

### 8. Why convert an image to RGB?

The model expects three color channels. Converting to RGB ensures a consistent color representation for images that may otherwise be grayscale or use another color mode.

### 9. What is normalization?

Normalization adjusts pixel values using predefined statistics, such as the ImageNet channel means and standard deviations, to match the preprocessing expected by the pretrained model.

### 10. What is Streamlit?

Streamlit is a Python framework for building interactive web applications, particularly for data science and machine learning projects.

### 11. What is PyTorch?

PyTorch is a deep learning framework used to create, train, and run neural networks. In this project, it performs model inference.

### 12. What is the top-5 prediction?

The top-5 prediction shows the five categories with the highest model output probabilities for an input image.

### 13. Does this project train the VGG-19 model?

No. The project uses pretrained ImageNet weights for inference and does not train the model on a custom dataset.

### 14. What is the purpose of softmax?

Softmax converts a vector of classification logits into normalized probability values. In this project, these values can be used to rank ImageNet categories.

### 15. What happens during the first run?

The pretrained model weights may be downloaded and cached locally. Once the weights are available, later runs can reuse the cached model.

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision VGG-19 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pillow Documentation](https://pillow.readthedocs.io/)

## 👨‍💻 Authors

### 1. Kattamuri Hemanth

- GitHub: [Hemu2174](https://github.com/Hemu2174)

### 2. CHOKKAPU DHANA LAKSHMI

- GitHub: [dhana3108](https://github.com/dhana3108)

---

⭐ If you find this project useful for learning computer vision and deep learning, consider starring the repository.

