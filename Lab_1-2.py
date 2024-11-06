import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math

class ReferenceImageForm(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Downloading reference images")
        self.geometry("600x800")  
        
       
        self.lab_title = tk.Label(self, text="Lab_2")
        self.lab_title.pack(pady=5) 
        
        # Оголошення класів і словників для зберігання зображень і векторів ознак
        self.class_labels = ["Клас 1", "Клас 2", "Клас 3"]
        self.reference_images = {label: [] for label in self.class_labels}
        self.normalized_feature_vectors = {label: [] for label in self.class_labels}
        self.karp_s1max = {label: [] for label in self.class_labels}
        self.karp_m1max = {label: [] for label in self.class_labels}
        self.karp_s1min = {label: [] for label in self.class_labels}  
        self.karp_m1min = {label: [] for label in self.class_labels}  

        self.unknown_image = None  
        self.unknown_feature_vector = None  
        
        # Поле введення для кількості еталонних зображень
        self.m_entry = tk.Entry(self)
        self.m_entry.insert(0, "5")  
        self.m_entry.pack()
        tk.Label(self, text="Кількість еталонних образів для кожного класу (m):").pack()
        
        # Поле введення для кількості секторів
        self.sector_count_entry = tk.Entry(self)
        self.sector_count_entry.insert(0, "4") 
        self.sector_count_entry.pack()
        tk.Label(self, text="Кількість секторів:").pack()

        for label in self.class_labels:
            button = tk.Button(self, text=f"Завантажити {label}", command=lambda l=label: self.load_reference_images(l))
            button.pack()

        self.load_unknown_button = tk.Button(self, text="Завантажити невідомий образ", command=self.load_unknown_image)
        self.load_unknown_button.pack()

        # Полотно для відображення зображень
        self.reference_canvas = tk.Canvas(self, width=580, height=350)
        self.reference_canvas.pack()

        self.calculate_button = tk.Button(self, text="Обчислити вектори ознак", command=self.calculate_feature_vectors)
        self.calculate_button.pack()

        self.classify_button = tk.Button(self, text="Класифікувати невідомий образ", command=self.classify_unknown_image)
        self.classify_button.pack()
        
        # Текстове поле для виведення результатів
        self.output_text = tk.Text(self, height=20, width=70)
        self.output_text.pack(pady=10)

    def load_reference_images(self, class_label):
        try:
            m = int(self.m_entry.get())
            if m <= 0:
                raise ValueError("Кількість образів повинна бути позитивним числом.")
        except ValueError as e:
            messagebox.showerror("Помилка", str(e))
            return

        # Завантаження кількох еталонних зображень
        file_paths = filedialog.askopenfilenames(
            title=f"Виберіть зображення для {class_label}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_paths:
            # Якщо вибрано більше зображень, ніж дозволяє кількість m, скоротимо список
            if len(file_paths) > m:
                file_paths = file_paths[:m]
            
            for file_path in file_paths:
                file_path = file_path.replace("\\", "/")
                image = cv2.imread(file_path)
                if image is not None:
                    self.reference_images[class_label].append(image)
                else:
                    messagebox.showerror("Помилка", f"Не вдалося завантажити зображення {file_path}.")
            
            self.display_reference_images()
        else:
            messagebox.showinfo("Інформація", f"Не вибрано зображень для {class_label}.")

    #                   messagebox.showerror("Помилка", "Не вдалося завантажити зображення.")

            self.display_reference_images()

    def load_unknown_image(self):
        # Викликати діалог вибору файлу для завантаження невідомого зображення
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        
        if file_path:
            # Завантажити зображення та конвертувати в формат OpenCV
            self.unknown_image = cv2.imread(file_path)
            if self.unknown_image is None:
                messagebox.showerror("Помилка", "Не вдалося завантажити зображення.")
                return

            # Очищення канви 
            self.reference_canvas.delete("all")  
            if hasattr(self.reference_canvas, 'images'):
                self.reference_canvas.images.clear()

            # Визначаємо розміри канви
            canvas_width = self.reference_canvas.winfo_width()
            canvas_height = self.reference_canvas.winfo_height()

            # Конвертувати зображення в формат для tkinter
            img_pil = Image.fromarray(cv2.cvtColor(self.unknown_image, cv2.COLOR_BGR2RGB))
            img_width, img_height = img_pil.size

            # Розрахувати координати, щоб розмістити зображення по центру канви
            x_position = (canvas_width - img_width) // 2
            y_position = (canvas_height - img_height) // 2

            # Конвертація зображення в формат для відображення в tkinter
            img_tk = ImageTk.PhotoImage(img_pil)

            # Відображення зображення по центру
            self.reference_canvas.create_image(x_position, y_position, anchor=tk.NW, image=img_tk)

            # Зберегти зображення у список, щоб його не було видалено зборщиком сміття
            if not hasattr(self.reference_canvas, 'images'):
                self.reference_canvas.images = []
            self.reference_canvas.images.append(img_tk)

    def display_reference_images(self, segmented=False):
        self.reference_canvas.delete("all")  

        img_width, img_height = 100, 100  
        x_offset = 10  
        y_offset = 10  

        for class_idx, (label, images) in enumerate(self.reference_images.items()):
            x_position = 0  
            y_position = class_idx * (img_height + y_offset) 

            for img in images:
                
                if segmented:
                    img_to_display = self.segment_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), int(self.sector_count_entry.get()))
                else:
                    img_to_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_pil = Image.fromarray(img_to_display)
                img_pil = img_pil.resize((img_width, img_height), Image.ANTIALIAS)
                img_tk = ImageTk.PhotoImage(img_pil)

                self.reference_canvas.create_image(x_position, y_position, anchor=tk.NW, image=img_tk)

                if not hasattr(self.reference_canvas, 'images'):
                    self.reference_canvas.images = []
                self.reference_canvas.images.append(img_tk)

                x_position += img_width + x_offset

        if self.unknown_image is not None:
            if segmented:
                img_to_display = self.segment_image(cv2.cvtColor(self.unknown_image, cv2.COLOR_BGR2GRAY), int(self.sector_count_entry.get()))
            else:
                img_to_display = cv2.cvtColor(self.unknown_image, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_to_display)
            img_pil = img_pil.resize((img_width, img_height), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img_pil)

            x_position = 10  
            y_position = len(self.class_labels) * (img_height + y_offset)  

            self.reference_canvas.create_image(x_position, y_position, anchor=tk.NW, image=img_tk)
            self.reference_canvas.images.append(img_tk)



    def calculate_feature_vectors(self):
        self.output_text.delete(1.0, tk.END)  
        try:
            sector_count = int(self.sector_count_entry.get())
            if sector_count <= 0:
                raise ValueError("Кількість секторів повинна бути позитивним числом.")
        except ValueError as e:
            messagebox.showerror("Помилка", str(e))
            return

        self.normalized_feature_vectors = {label: [] for label in self.class_labels}

        for label, images in self.reference_images.items():
            for img in images:
                threshold = 127  
                binary_image = self.convert_to_binary(img, threshold)
                
                segmented_image = self.segment_image(binary_image, sector_count)

                feature_vector = self.calculate_feature_vector(binary_image, sector_count)
                normalized_s1, normalized_m1 = self.normalize_feature_vector(feature_vector)
                self.normalized_feature_vectors[label].append((normalized_s1, normalized_m1))

                output = f"{label}:\n"
                output += f"  Absolute Feature Vector: {feature_vector}\n"
                output += f"  Normalized S1: {normalized_s1}\n"
                output += f"  Normalized M1: {normalized_m1}\n\n"
                self.output_text.insert(tk.END, output)

            if self.normalized_feature_vectors[label]:
                max_s1 = np.max([vec[0] for vec in self.normalized_feature_vectors[label]], axis=0)
                max_m1 = np.max([vec[1] for vec in self.normalized_feature_vectors[label]], axis=0)
                self.karp_s1max[label] = max_s1.tolist()
                self.karp_m1max[label] = max_m1.tolist()

                min_s1 = np.min([vec[0] for vec in self.normalized_feature_vectors[label]], axis=0)
                min_m1 = np.min([vec[1] for vec in self.normalized_feature_vectors[label]], axis=0)
                self.karp_s1min[label] = min_s1.tolist()
                self.karp_m1min[label] = min_m1.tolist()

                output += f"{label} Karp S1MAX: {self.karp_s1max[label]}\n"
                output += f"{label} Karp M1MAX: {self.karp_m1max[label]}\n"
                output += f"{label} Karp S1MIN: {self.karp_s1min[label]}\n"
                output += f"{label} Karp M1MIN: {self.karp_m1min[label]}\n\n"
                self.output_text.insert(tk.END, output)

        self.display_reference_images(segmented=True)

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

    def classify_unknown_image(self):
        if self.unknown_image is None:
            messagebox.showerror("Помилка", "Спочатку завантажте невідомий образ.")
            return

        try:
            sector_count = int(self.sector_count_entry.get())
            if sector_count <= 0:
                raise ValueError("Кількість секторів повинна бути позитивним числом.")
        except ValueError as e:
            messagebox.showerror("Помилка", str(e))
            return

        threshold = 127  
        binary_image = self.convert_to_binary(self.unknown_image, threshold)

        feature_vector = self.calculate_feature_vector(binary_image, sector_count)

        normalized_s1, normalized_m1 = self.normalize_feature_vector(feature_vector)

        for label in self.class_labels:
            if len(normalized_s1) != len(self.karp_s1min[label]) or len(normalized_s1) != len(self.karp_s1max[label]):
                messagebox.showerror("Помилка", f"Невідповідність довжин векторів для класу {label}.")
                return

        matching_classes = []

        for label in self.class_labels:
            if all(self.karp_s1min[label][i] <= normalized_s1[i] <= self.karp_s1max[label][i] for i in range(len(normalized_s1))):
                matching_classes.append(label)

        self.output_text.delete(1.0, tk.END)
        if len(matching_classes) == 0:
            classification_result = "Невідомий образ не належить жодному класу."
        elif len(matching_classes) == 1:
            classification_result = f"Невідомий образ належить класу {matching_classes[0]}."
        else:
            classification_result = f"Невідомий образ належить до наступних класів: {', '.join(matching_classes)}."

        self.output_text.insert(tk.END, f"Результат класифікації: {classification_result}\n")
        self.output_text.insert(tk.END, f"\n{label} Karp S1MIN: {self.karp_s1min[label]}\n")
        self.output_text.insert(tk.END, f"{label} Karp S1MAX: {self.karp_s1max[label]}\n")
            
    def convert_to_binary(self, image, threshold):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def calculate_feature_vector(self, image, sector_count):
        height, width = image.shape
        feature_vector = [] 
        
        angles = np.linspace(0, 90, sector_count + 1)[1:]  

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
        normalized_s1 = feature_vector / np.linalg.norm(feature_vector)
        normalized_m1 = (feature_vector - np.min(feature_vector)) / (np.max(feature_vector) - np.min(feature_vector))

        return normalized_s1, normalized_m1

    
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

        self.crop_enabled = False
        self.start_x = None
        self.start_y = None
        self.crop_rectangle = None
        self.reference_button = tk.Button(root, text="Відкрити форму еталонних образів", command=self.open_reference_form)
        self.reference_button.pack()
        
    def open_reference_form(self):
        ReferenceImageForm(self.root)
        
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

        self.scaled_image = image_pil.resize(
            (int(image_pil.width * self.zoom_scale), int(image_pil.height * self.zoom_scale)), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(self.scaled_image)

        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width, image_height = image_tk.width(), image_tk.height()

        self.image_offset_x = (canvas_width - image_width) // 2
        self.image_offset_y = (canvas_height - image_height) // 2

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
                crop_start_x = max(0, (self.start_x - self.image_offset_x) / self.zoom_scale)
                crop_start_y = max(0, (self.start_y - self.image_offset_y) / self.zoom_scale)
                crop_end_x = max(0, (end_x - self.image_offset_x) / self.zoom_scale)
                crop_end_y = max(0, (end_y - self.image_offset_y) / self.zoom_scale)

                crop_start_x = int(min(self.image.shape[1], crop_start_x))
                crop_start_y = int(min(self.image.shape[0], crop_start_y))
                crop_end_x = int(min(self.image.shape[1], crop_end_x))
                crop_end_y = int(min(self.image.shape[0], crop_end_y))

                self.image = self.image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

                self.canvas.delete(self.crop_rectangle)
                self.crop_rectangle = None

                self.show_image(self.image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()