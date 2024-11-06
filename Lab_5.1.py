import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        
    def predict(self, features):
        return 1 if np.dot(features, self.weights) + self.bias > 0 else 0
    
    def train(self, features, label):
        prediction = self.predict(features)
        error = label - prediction
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error
        return error

class ImageClassifier:
    def __init__(self):
        self.perceptron = None
        self.feature_vectors = []
        self.labels = []
        self.num_segments = 5
        self.threshold = 150
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Perceptron Image Classifier")
        self.root.geometry("800x600")
        
        # Створення фреймів
        self.training_frame = ttk.LabelFrame(self.root, text="Навчання")
        self.training_frame.pack(padx=10, pady=5, fill="x")
        
        self.params_frame = ttk.LabelFrame(self.root, text="Параметри")
        self.params_frame.pack(padx=10, pady=5, fill="x")
        
        self.classification_frame = ttk.LabelFrame(self.root, text="Класифікація")
        self.classification_frame.pack(padx=10, pady=5, fill="x")
        
        # Параметри перцептрона
        ttk.Label(self.params_frame, text="Швидкість навчання:").grid(row=0, column=0, padx=5, pady=5)
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.learning_rate_entry = ttk.Entry(self.params_frame, textvariable=self.learning_rate_var)
        self.learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.params_frame, text="Поріг:").grid(row=1, column=0, padx=5, pady=5)
        self.threshold_var = tk.StringVar(value="150")
        self.threshold_entry = ttk.Entry(self.params_frame, textvariable=self.threshold_var)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.params_frame, text="Сегменти:").grid(row=2, column=0, padx=5, pady=5)
        self.segments_var = tk.StringVar(value="4")
        self.segments_entry = ttk.Entry(self.params_frame, textvariable=self.segments_var)
        self.segments_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Кнопки навчання
        ttk.Button(self.training_frame, text="Class 1", command=lambda: self.load_training_images(1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.training_frame, text="Class 2", command=lambda: self.load_training_images(0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.training_frame, text="Ініціалізація Perceptron", command=self.initialize_perceptron).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.training_frame, text="Тренувати Perceptron", command=self.train_perceptron).pack(side=tk.LEFT, padx=5)
        
        # Класифікація
        ttk.Button(self.classification_frame, text="Завантажити зображення", 
                  command=self.load_classification_image).pack(pady=5)
        
        # Результати
        self.result_text = tk.Text(self.root, height=20, width=80)
        self.result_text.pack(padx=5, pady=5)
        
    def process_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, thresh_image = cv2.threshold(image, int(self.threshold_var.get()), 255, cv2.THRESH_BINARY)
        return self.crop_image(thresh_image)
    
    def crop_image(self, image):
        coordinates = np.column_stack(np.where(image == 0))
        if coordinates.size == 0:
            return image
        top_left = coordinates.min(axis=0)
        bottom_right = coordinates.max(axis=0)
        return image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    
    def calculate_feature_vector(self, image):
        height, width = image.shape
        num_segments = int(self.segments_var.get())
        feature_vector = []
        angles = np.linspace(0, 90, num_segments + 1)[1:]
        
        for angle in angles:
            rad_angle = np.radians(angle)
            x_start = width
            y_start = height
            x_end = 0
            y_end = int((width - x_end) * np.tan(rad_angle))
            
            if y_end > height:
                y_end = height
                x_end = width - int(height / np.tan(rad_angle))
            
            mask = np.zeros((height, width), dtype=np.uint8)
            polygon_points = np.array([[width, height], [x_end, height - y_end], 
                                     [0, 0], [width, 0]], np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon_points], 255)
            
            masked_image = cv2.bitwise_and(image, mask)
            black_pixels = np.sum(masked_image == 0)
            feature_vector.append(black_pixels)
            
        return self.normalize_vector(feature_vector)
    
    def normalize_vector(self, vector):
        total = sum(vector)
        return np.array([x/total if total > 0 else 0 for x in vector])
    
    def load_training_images(self, label):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        
        for file_path in file_paths:
            image = self.process_image(file_path)
            features = self.calculate_feature_vector(image)
            self.feature_vectors.append(features)
            self.labels.append(label)
            
            self.result_text.insert(tk.END, 
                f"Завантажене зображення для класу {label}: {os.path.basename(file_path)}\n")
            self.result_text.insert(tk.END, f"Нормований вектор: {features}\n\n")
            self.result_text.see(tk.END)
    
    def initialize_perceptron(self):
        try:
            num_segments = int(self.segments_var.get())
            learning_rate = float(self.learning_rate_var.get())
            self.perceptron = Perceptron(num_segments, learning_rate)
            
            self.result_text.insert(tk.END, 
                f"Ініціалізований перцептрон з вагами: {self.perceptron.weights}\n")
            self.result_text.insert(tk.END, f"Bias: {self.perceptron.bias}\n\n")
            self.result_text.see(tk.END)
        except ValueError as e:
            messagebox.showerror("Error", "Недійсні параметри")
    
    def train_perceptron(self):
        if not self.perceptron or not self.feature_vectors:
            messagebox.showerror("Error", "Спочатку ініціалізуйте персептрон і завантажте навчальні дані")
            return
            
        epochs = 100
        for epoch in range(epochs):
            total_error = 0
            for features, label in zip(self.feature_vectors, self.labels):
                error = self.perceptron.train(features, label)
                total_error += abs(error)
            
            if total_error == 0:
                self.result_text.insert(tk.END, 
                    f"Навчання зійшлося в епоху {epoch + 1}\n")
                break
                
            if epoch % 10 == 0:
                self.result_text.insert(tk.END, 
                    f"Епоха {epoch + 1}, Total error: {total_error}\n")
                self.result_text.see(tk.END)
        
        self.result_text.insert(tk.END, 
            f"Остаточні ваги: {self.perceptron.weights}\n")
        self.result_text.insert(tk.END, f"Остаточний ухил: {self.perceptron.bias}\n\n")
        self.result_text.see(tk.END)
    
    def load_classification_image(self):
        if not self.perceptron:
            messagebox.showerror("Error", "Спочатку навчіть перцептрон")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            image = self.process_image(file_path)
            features = self.calculate_feature_vector(image)
            prediction = self.perceptron.predict(features)
            
            self.result_text.insert(tk.END, 
                f"Результат класифікації для {os.path.basename(file_path)}:\n")
            self.result_text.insert(tk.END, 
                f"Class: {prediction} (0: Class 2, 1: Class 1)\n")
            self.result_text.insert(tk.END, f"Нормований вектор: {features}\n\n")
            self.result_text.see(tk.END)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    classifier = ImageClassifier()
    classifier.run()