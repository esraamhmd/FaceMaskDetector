# Face Mask Detection System

A comprehensive face mask detection system built with deep learning that can classify images of faces to determine mask-wearing status. The system uses transfer learning with MobileNetV3 architecture to efficiently identify whether individuals are wearing masks properly, improperly, or not at all.

## 🎯 Project Overview

This project implements a face mask detection system using advanced deep learning techniques, particularly relevant for public health monitoring and mask compliance verification. The system provides real-time detection capabilities through an intuitive GUI interface built with Tkinter.



https://github.com/user-attachments/assets/d0672342-9bea-4477-8219-08842e989da6




## ✨ Features

- **Binary Classification**: Accurately detects "with_mask" vs "without_mask" status
- **High Performance**: Achieves 100% accuracy on test dataset
- **Real-time Detection**: Interactive GUI for immediate image analysis
- **Transfer Learning**: Leverages pre-trained MobileNetV3 for efficient training
- **User-friendly Interface**: Simple Tkinter-based GUI with visual feedback
- **Robust Preprocessing**: Comprehensive data augmentation and preprocessing pipeline

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework for model development and training
- **Kaggle Environment**: Cloud platform for dataset access and model training
- **Tkinter**: Python GUI framework for desktop application interface
- **OpenCV**: Image processing and computer vision operations
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and result plotting

## 🏗️ Model Architecture

### MobileNetV3 Base Architecture

The system utilizes **MobileNetV3** as the backbone network, chosen for its optimal balance between accuracy and computational efficiency.

![image](https://github.com/user-attachments/assets/2e0c0dfb-caf2-4cb7-b5bd-70b0762e201a)


#### Key Components:

1. **Input Layer**
   - Accepts RGB images (224×224×3)
   - Standardized input format for consistent processing

2. **Initial Convolutional Layer**
   - Standard convolution (not depthwise separable)
   - 3×3 kernel with stride 2
   - Batch normalization and ReLU6 activation

3. **Bottleneck Blocks**
   - Core building blocks with three operations:
     - 1×1 expansion convolution
     - 3×3 depthwise convolution (spatial filtering)
     - 1×1 projection convolution
   - Residual connections for gradient flow
   - ReLU6 activation functions

4. **Global Average Pooling**
   - Reduces spatial dimensions to 1×1
   - Minimizes overfitting risk
   - Efficient parameter reduction

5. **Fully Connected Layer**
   - Dense layer for feature mapping
   - Connects to binary classification output

6. **Softmax Layer**
   - Probability distribution output
   - Binary classification: with_mask / without_mask

## 📊 System Workflow

```
Input Image → Preprocessing → Dataset Partitioning → Training/Validation/Testing
                                       ↓
Classification ← Trained Model ← MobileNetV3 Training
```
![image](https://github.com/user-attachments/assets/56857ba7-76b1-4698-89e9-2c208c47c7da)

### Processing Pipeline:

1. **Data Preprocessing**: Image normalization, resizing, and augmentation
2. **Dataset Partitioning**: Train/validation/test split for robust evaluation
3. **Model Training**: Transfer learning with MobileNetV3 base
4. **Validation**: Performance monitoring during training
5. **Testing**: Final model evaluation on unseen data
6. **Classification**: Real-time prediction on new images

## 📈 Performance Metrics

### Classification Results:
- **Overall Accuracy**: 100%
- **Precision**: 
  - With Mask: 99%
  - Without Mask: 100%
- **Recall**: 
  - With Mask: 100%
  - Without Mask: 99%
- **F1-Score**: 100% for both classes

### Confusion Matrix:
- True Positives (with_mask): 572
- True Negatives (without_mask): 543
- False Positives: 2
- False Negatives: 3
- **Total Test Samples**: 1,120

## 🖥️ GUI Interface

The system features an intuitive Tkinter-based desktop application with:

### Key Features:
- **Image Selection**: Browse and load images for analysis
- **Real-time Detection**: Instant mask detection with confidence scores
- **Visual Feedback**: 
  - Red bounding box for "without_mask" (100% confidence)
  - Green bounding box for "with_mask" (100% confidence)
- **Detection Statistics**: 
  - Total faces detected
  - Percentage breakdown of mask/no-mask detection
  - Processing status indicators

### Interface Components:
- Model loading section with file path display
- Image display area with detection overlays
- Results panel showing detection statistics
- Control buttons for image selection and detection execution

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/esraamhmd/FaceMaskDetector.git
   cd FaceMaskDetector
   ```

2. **Install Dependencies**
   ```bash
   pip install tensorflow opencv-python tkinter numpy matplotlib pillow
   ```

3. **Download Pre-trained Model**
   - Ensure the trained model file (`face_mask_mobilenetV3.h5`) is in the project directory
   - Model can be downloaded from the releases section or trained using the provided notebook

## 💻 Usage

### Running the GUI Application:
```bash
python face_mask_detector_gui.py
```

### Using the Interface:
1. **Load Model**: Click "Load Model" to initialize the pre-trained model
2. **Select Image**: Choose an image file for mask detection
3. **Detect Masks**: Click "Detect Masks" to analyze the selected image
4. **View Results**: Check detection results in the statistics panel

### Supported Image Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 📁 Project Structure

```
FaceMaskDetector/
├── face_mask_detector_gui.py    # Main GUI application
├── model_training.ipynb         # Training notebook
├── face_mask_mobilenetV3.h5     # Trained model weights
├── requirements.txt             # Project dependencies
├── README.md                   # Project documentation
└── sample_images/              # Test images directory
```

## 🔬 Model Training

The model training process includes:

1. **Data Augmentation**: Rotation, scaling, brightness adjustment
2. **Transfer Learning**: Fine-tuning MobileNetV3 pre-trained weights
3. **Optimization**: Adam optimizer with learning rate scheduling
4. **Regularization**: Dropout and batch normalization for generalization
5. **Early Stopping**: Prevents overfitting during training

## 🎯 Applications

- **Public Health Monitoring**: Automated mask compliance checking
- **Security Systems**: Integration with surveillance cameras
- **Educational Tools**: Demonstration of computer vision concepts
- **Research Platform**: Base for advanced face detection research

## 🔧 Customization

The system can be extended for:
- **Multi-class Detection**: Proper/improper mask wearing classification
- **Real-time Video**: Webcam integration for live detection
- **Mobile Deployment**: Conversion to mobile-friendly formats
- **API Integration**: RESTful service for web applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Esraa Mohammed** - [GitHub Profile](https://github.com/esraamhmd)

## 🙏 Acknowledgments

- TensorFlow team for the MobileNetV3 architecture
- Kaggle community for dataset resources and compute environment
- OpenCV contributors for computer vision tools
- Public health organizations inspiring this COVID-19 era project

## 📞 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Contact the repository owner through GitHub

---

⭐ **Star this repository if you find it helpful for your projects!**
