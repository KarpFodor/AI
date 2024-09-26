import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Feature Vector Builder")
        
        self.label = tk.Label(root, text="Lab_1")
        self.label.pack()

        self.load_button = tk.Button(root, text="Завантажити зображення", command=self.load_image)
        self.load_button.pack()

        self.threshold_slider = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Порогове значення")
        self.threshold_slider.set(127)
        self.threshold_slider.pack()

        self.sector_count_entry = tk.Entry(root)
        self.sector_count_entry.insert(0, "4")  
        self.sector_count_entry.pack()
        tk.Label(root, text="Кількість секторів:").pack()

        self.process_button = tk.Button(root, text="Обробити зображення", command=self.process_image)
        self.process_button.pack()

        self.crop_button = tk.Button(root, text="Обрізати зображення", command=self.enable_crop)
        self.crop_button.pack()

        self.canvas = tk.Canvas(root, width=500, height=400)
        self.canvas.pack()

        self.image = None
        self.file_path = None

        self.zoom_scale = 1.0
        self.canvas.bind("<MouseWheel>", self.zoom_image)

        # Variables for cropping
        self.crop_enabled = False
        self.start_x = None
        self.start_y = None
        self.crop_rectangle = None

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        
        if self.file_path:
            self.file_path = self.file_path.replace("\\", "/")
            self.image = cv2.imread(self.file_path)

            if self.image is not None:
                self.show_image(self.image)
            else:
                messagebox.showerror("Помилка", "Не вдалося завантажити зображення. Перевірте файл і шлях до нього.")
        else:
            messagebox.showinfo("Інформація", "Файл не вибрано")

    def process_image(self):
        if self.image is not None:
            threshold = self.threshold_slider.get()
            binary_image = self.convert_to_binary(self.image, threshold)
            localized_image = self.localize_image(binary_image)
            
            sector_count = int(self.sector_count_entry.get())
            segmented_image = self.segment_image(localized_image, sector_count)
            feature_vector = self.calculate_feature_vector(localized_image, sector_count)

            normalized_s1, normalized_m1 = self.normalize_feature_vector(feature_vector)
            KarpS1, KarpM1 = normalized_s1, normalized_m1
            print("Feature Vector:", feature_vector)
            print("Karp S1 :", KarpS1)
            print("Karp M1 :", KarpM1)

            self.show_image(segmented_image)
        else:
            messagebox.showwarning("Увага", "Зображення не завантажене. Спочатку завантажте зображення.")

    def show_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Масштабируем изображение в соответствии с текущим zoom_scale
        self.scaled_image = image_pil.resize(
            (int(image_pil.width * self.zoom_scale), int(image_pil.height * self.zoom_scale)), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(self.scaled_image)

        # Очищаем холст
        self.canvas.delete("all")

        # Вычисляем сдвиг для центрирования изображения
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width, image_height = image_tk.width(), image_tk.height()

        self.image_offset_x = (canvas_width - image_width) // 2
        self.image_offset_y = (canvas_height - image_height) // 2

        # Отображаем изображение на холсте с учетом сдвига
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, anchor=tk.NW, image=image_tk)
        self.canvas.image = image_tk
        
    def convert_to_binary(self, image, threshold):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def localize_image(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            localized = image[y:y+h, x:x+w]
            return localized
        return image

    def segment_image(self, image, sector_count):
        height, width = image.shape
        center_x, center_y = width - 1, height - 1  
        radius = min(width, height)

        segmented_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        angle_step = 90 / sector_count  

        for i in range(sector_count):
            angle = i * angle_step
            x_end = int(center_x - radius * math.cos(math.radians(angle)))
            y_end = int(center_y - radius * math.sin(math.radians(angle)))
            cv2.line(segmented_image, (center_x, center_y), (x_end, y_end), (0, 255, 0), 1)

        return segmented_image

    def calculate_feature_vector(self, image, sector_count):
        height, width = image.shape
        feature_vector = [] 
        
        angles = np.linspace(0, 90, sector_count + 1)
        angles = angles[1:]  

        for angle in angles:
            rad_angle = np.radians(angle)
        
            x_end = width
            y_end = int(x_end * np.tan(rad_angle))
        
            if y_end > height:
                 y_end = height
                 x_end = int(y_end / np.tan(rad_angle))
        
            mask = np.zeros_like(image, dtype=np.uint8)
            
            polygon_points = np.array([
                [0, 0],
                [x_end, y_end],
                [width, height],
                [0, height]
            ], np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
                
            cv2.fillPoly(mask, [polygon_points], 255)
            
            masked_image = np.zeros((height, width), dtype=np.uint8)
            for x in range(width):
                for y in range(height):
                    if image[y, x] == mask[y, x] and image[y, x] == 0:
                     masked_image[y, x] = 1
                     
            count = np.sum(masked_image == 1)
            previous_sum = sum(feature_vector)
            black_pixels = count - previous_sum
            feature_vector.append(black_pixels)
        
        return feature_vector

    def normalize_feature_vector(self, feature_vector):
        feature_vector = np.array(feature_vector)
        norm_s1 = feature_vector / np.linalg.norm(feature_vector)
        norm_m1 = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector))
        return norm_s1.tolist(), norm_m1.tolist()

    def zoom_image(self, event):
        if event.delta > 0:
            self.zoom_scale *= 1.1  
        else:
            self.zoom_scale /= 1.1 
        if self.image is not None:
            self.show_image(self.image)

    def enable_crop(self):
        self.crop_enabled = True
        self.canvas.bind("<ButtonPress-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.perform_crop)

    def start_crop(self, event):
        if self.crop_enabled:
            self.start_x, self.start_y = event.x, event.y
            self.crop_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def update_crop(self, event):
        if self.crop_enabled and self.crop_rectangle:
            self.canvas.coords(self.crop_rectangle, self.start_x, self.start_y, event.x, event.y)

    def perform_crop(self, event):
        if self.crop_enabled:
            self.crop_enabled = False
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

            end_x, end_y = event.x, event.y
            if self.image is not None:
                # Пересчитываем координаты выделения относительно изображения, а не холста
                crop_start_x = max(0, (self.start_x - self.image_offset_x) / self.zoom_scale)
                crop_start_y = max(0, (self.start_y - self.image_offset_y) / self.zoom_scale)
                crop_end_x = max(0, (end_x - self.image_offset_x) / self.zoom_scale)
                crop_end_y = max(0, (end_y - self.image_offset_y) / self.zoom_scale)

                crop_start_x = int(min(self.image.shape[1], crop_start_x))
                crop_start_y = int(min(self.image.shape[0], crop_start_y))
                crop_end_x = int(min(self.image.shape[1], crop_end_x))
                crop_end_y = int(min(self.image.shape[0], crop_end_y))

                # Обрезаем изображение
                self.image = self.image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

                # Удаляем красный прямоугольник с холста
                self.canvas.delete(self.crop_rectangle)
                self.crop_rectangle = None

                # Отображаем обрезанное изображение
                self.show_image(self.image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
