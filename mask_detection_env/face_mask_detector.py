import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

class FaceMaskDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Model variables
        self.model = None
        self.model_loaded = False
        self.class_names = {0: "with_mask", 1: "without_mask"}
        self.IMG_SIZE = 224
        
        # Create UI
        self.create_ui()
        
        # Try to load the model
        self.load_model_button_click()
    
    def create_ui(self):
        # Create frames
        self.header_frame = tk.Frame(self.root, bg="#3498db", height=60)
        self.header_frame.pack(fill=tk.X)
        
        self.content_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.left_frame = tk.Frame(self.content_frame, bg="#f0f0f0", width=600)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = tk.Frame(self.content_frame, bg="#f0f0f0", width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Header elements
        tk.Label(
            self.header_frame, 
            text="Face Mask Detection System", 
            font=("Arial", 18, "bold"),
            bg="#3498db",
            fg="white"
        ).pack(pady=10)
        
        # Left frame - Image display
        self.image_frame = tk.Frame(self.left_frame, bg="white", height=500, width=500)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image selection buttons
        self.button_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        self.button_frame.pack(fill=tk.X)
        
        # Changed button color from #2ecc71 to #3498db (blue)
        self.select_image_btn = tk.Button(
            self.button_frame,
            text="Select Image",
            font=("Arial", 12),
            bg="#3498db",  # Changed from green to blue
            fg="white",
            padx=15,
            pady=8,
            command=self.select_image
        )
        self.select_image_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Changed button color from #e74c3c to #f39c12 (orange)
        self.process_image_btn = tk.Button(
            self.button_frame,
            text="Detect Masks",
            font=("Arial", 12),
            bg="#f39c12",  # Changed from red to orange
            fg="white",
            padx=15,
            pady=8,
            command=self.process_image,
            state=tk.DISABLED
        )
        self.process_image_btn.pack(side=tk.LEFT)
        
        # Right frame - Controls and Results
        # Model loading
        self.model_frame = tk.LabelFrame(self.right_frame, text="Model", bg="#f0f0f0", font=("Arial", 12))
        self.model_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.model_path_var = tk.StringVar(value="face_mask_mobilenetv2.h5")
        self.model_entry = tk.Entry(self.model_frame, textvariable=self.model_path_var, width=25)
        self.model_entry.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.load_model_btn = tk.Button(
            self.model_frame,
            text="Load Model",
            command=self.load_model_button_click
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Results frame
        self.results_frame = tk.LabelFrame(self.right_frame, text="Detection Results", bg="#f0f0f0", font=("Arial", 12))
        self.results_frame.pack(fill=tk.X)
        
        # Status indicator
        self.status_frame = tk.Frame(self.results_frame, bg="#f0f0f0")
        self.status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(self.status_frame, text="Status:", bg="#f0f0f0", font=("Arial", 12)).pack(side=tk.LEFT)
        self.status_label = tk.Label(
            self.status_frame, 
            text="Ready", 
            bg="#f0f0f0", 
            fg="#2ecc71",
            font=("Arial", 12, "bold")
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Statistics
        self.stats_frame = tk.Frame(self.results_frame, bg="#f0f0f0")
        self.stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Total faces
        tk.Label(self.stats_frame, text="Total Faces:", bg="#f0f0f0", font=("Arial", 11)).grid(row=0, column=0, sticky="w", pady=5)
        self.total_faces_var = tk.StringVar(value="0")
        tk.Label(self.stats_frame, textvariable=self.total_faces_var, bg="#f0f0f0", font=("Arial", 11, "bold")).grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # With mask
        tk.Label(self.stats_frame, text="With Mask:", bg="#f0f0f0", font=("Arial", 11)).grid(row=1, column=0, sticky="w", pady=5)
        self.with_mask_var = tk.StringVar(value="0 (0.00%)")
        tk.Label(self.stats_frame, textvariable=self.with_mask_var, bg="#f0f0f0", font=("Arial", 11, "bold"), fg="#2ecc71").grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        # Without mask
        tk.Label(self.stats_frame, text="Without Mask:", bg="#f0f0f0", font=("Arial", 11)).grid(row=2, column=0, sticky="w", pady=5)
        self.without_mask_var = tk.StringVar(value="0 (0.00%)")
        tk.Label(self.stats_frame, textvariable=self.without_mask_var, bg="#f0f0f0", font=("Arial", 11, "bold"), fg="#e74c3c").grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        # Add progress bar
        self.progress_frame = tk.Frame(self.right_frame, bg="#f0f0f0")
        self.progress_frame.pack(fill=tk.X, pady=20)
        
        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X)
        
        # Footer with instructions
        self.footer_frame = tk.Frame(self.root, bg="#ecf0f1", height=40)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(
            self.footer_frame,
            text="1. Load the model  2. Select an image  3. Click 'Detect Masks'",
            font=("Arial", 10),
            bg="#ecf0f1", 
            fg="#7f8c8d"
        ).pack(pady=10)
        
        # Variables
        self.image_path = None
        self.detected_image = None
    
    def load_model_button_click(self):
        # Get the model path from the entry
        model_path = self.model_path_var.get()
        
        # Update status
        self.status_label.config(text="Loading model...", fg="orange")
        self.progress.start()
        self.root.update()
        
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
            self.status_label.config(text="Model loaded successfully!", fg="#2ecc71")
            messagebox.showinfo("Success", "Face mask detection model loaded successfully!")
        except Exception as e:
            self.model_loaded = False
            self.status_label.config(text="Failed to load model", fg="#e74c3c")
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
        
        self.progress.stop()
    
    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if self.image_path:
            # Display the selected image
            self.display_image(self.image_path)
            
            # Enable the process button if the model is loaded
            if self.model_loaded:
                self.process_image_btn.config(state=tk.NORMAL)
    
    def display_image(self, image_path, processed=False):
        try:
            # Open the image using PIL
            img = Image.open(image_path)
            
            # Calculate the resize ratio to fit the frame
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()
            
            # Resize image to fit the frame while maintaining aspect ratio
            img_width, img_height = img.size
            ratio = min(frame_width/img_width, frame_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update the image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  
            
            if not processed:
                self.status_label.config(text="Image loaded, ready to detect", fg="blue")
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image: {str(e)}")
    
    def process_image(self):
        if not self.image_path or not self.model_loaded:
            messagebox.showwarning("Warning", "Please select an image and load a model first.")
            return
        
        self.status_label.config(text="Processing...", fg="orange")
        self.progress.start()
        self.root.update()
        
        try:
            # Read image with OpenCV
            img = cv2.imread(self.image_path)
            
            # Convert to RGB (from BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load face detector
            face_cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(face_cascade_file):
                raise Exception(f"Face cascade file not found at: {face_cascade_file}")
                
            face_cascade = cv2.CascadeClassifier(face_cascade_file)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                self.status_label.config(text="No faces detected", fg="#e74c3c")
                self.progress.stop()
                
                # Update stats
                self.total_faces_var.set("0")
                self.with_mask_var.set("0 (0.00%)")
                self.without_mask_var.set("0 (0.00%)")
                
                # Show the original image
                self.display_image(self.image_path)
                
                messagebox.showinfo("Result", "No faces detected in the image.")
                return
            
            # Initialize counts
            mask_count = 0
            no_mask_count = 0
            
            # Loop through detected faces
            for (x, y, w, h) in faces:
                face_roi = img_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (self.IMG_SIZE, self.IMG_SIZE))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                prediction = self.model.predict(face_input)
                predicted_index = np.argmax(prediction[0])
                predicted_label = self.class_names[predicted_index]
                confidence = prediction[0][predicted_index]
                
                if predicted_label == "with_mask":
                    mask_count += 1
                    color = (0, 255, 0)  # Green for masked
                else:
                    no_mask_count += 1
                    color = (255, 0, 0)  # Red for unmasked
                
                # Updated to show confidence as percentage without decimal places
                label_text = f"{predicted_label}: {confidence * 100:.0f}%"
                
                # Draw bounding box
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
                
                # Calculate label size
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Coordinates for label box
                label_x = x
                label_y = y - 10 if y - 10 > 10 else y + text_height + 10
                
                # Draw filled background for label
                cv2.rectangle(img_rgb, (label_x, label_y - text_height - 5),
                            (label_x + text_width + 10, label_y + 5), color, -1)
                
                # Put label text
                cv2.putText(img_rgb, label_text, (label_x + 5, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert back to PIL format for display
            processed_img = Image.fromarray(img_rgb)
            
            # Save the processed image temporarily
            temp_output_path = "temp_processed_image.png"
            processed_img.save(temp_output_path)
            
            # Display the processed image
            self.display_image(temp_output_path, processed=True)
            
            # Stats
            total_faces = mask_count + no_mask_count
            mask_percent = (mask_count / total_faces) * 100 if total_faces > 0 else 0
            no_mask_percent = (no_mask_count / total_faces) * 100 if total_faces > 0 else 0
            
          
            self.total_faces_var.set(str(total_faces))
            self.with_mask_var.set(f"{mask_count} ({mask_percent:.0f}%)")
            self.without_mask_var.set(f"{no_mask_count} ({no_mask_percent:.0f}%)")
            
            self.status_label.config(text="Detection completed", fg="#2ecc71")
            
            # Print detection summary 
            print("Detection Summary:")
            print(f"Total Faces: {total_faces}")
            print(f"With Mask: {mask_count} ({mask_percent:.0f}%)")
            print(f"Without Mask: {no_mask_count} ({no_mask_percent:.0f}%)")
            
        except Exception as e:
            self.status_label.config(text="Error processing image", fg="#e74c3c")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
        
        self.progress.stop()

def main():
    # Create Tkinter root window
    root = tk.Tk()
    
    # Create the app
    app = FaceMaskDetectorApp(root)
    
    # Start the mainloop
    root.mainloop()

if __name__ == "__main__":
    main()