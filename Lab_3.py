import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class RecognitionSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Система Розпізнавання")
        self.master.geometry("1000x700")
        self.master.configure(bg="#0850bc")

        self.class_images = [[] for _ in range(3)]  # Образи трьох класів
        self.feature_vectors = [[] for _ in range(3)]
        self.centroids = [None, None, None]

        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title_label = Label(self.master, text="Lab_3", font=("Helvetica", 16, "bold"), fg="#ecf0f1", bg="#0850bc")
        title_label.pack(pady=20)

        # Ліва панель з кнопками
        button_frame = Frame(self.master, bg="#20dbd8")
        button_frame.pack(side=LEFT, padx=20)

        for i in range(3):
            btn = Button(button_frame, text=f"Вибрати образи класу {i + 1}", command=lambda i=i: self.load_images(i),
                         bg="#1da3bb", fg="white", font=("Helvetica", 12), relief=FLAT)
            btn.pack(pady=5, padx=2)

            view_btn = Button(button_frame, text=f"Переглянути образи класу {i + 1}", command=lambda i=i: self.view_class_images(i),
                              bg="#146c86", fg="white", font=("Helvetica", 12), relief=FLAT)
            view_btn.pack(pady=10, padx=10)

        self.unknown_btn = Button(button_frame, text="Вибрати невідомий образ", command=self.load_unknown_image,
                                  bg="#124155", fg="white", font=("Helvetica", 12), relief=FLAT)
        self.unknown_btn.pack(pady=20, padx=2)

        # Поле для відображення результатів
        self.result_frame = Frame(self.master, bg="#0850bc")
        self.result_frame.pack(side=RIGHT, padx=20, pady=10, fill=Y)

        self.result_label = Label(self.result_frame, text="Результати класифікації", font=("Helvetica", 14), bg="#0850bc", fg="white", wraplength=300)
        self.result_label.pack(pady=20)

        # Поле для відображення зображень
        self.image_label = Label(self.master, bg="#ecf0f1", borderwidth=2, relief="groove")
        self.image_label.pack(side=RIGHT, padx=10, pady=5, fill=BOTH, expand=True)

    def load_images(self, class_index):
        file_paths = filedialog.askopenfilenames(title="Виберіть зображення", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if len(file_paths) < 5:
            messagebox.showerror("Помилка", "Виберіть не менше 5 зображень.")
            return

        self.class_images[class_index] = [cv2.imread(path) for path in file_paths]
        self.feature_vectors[class_index] = [self.extract_features(image) for image in self.class_images[class_index]]
        self.centroids[class_index] = self.compute_centroid(self.feature_vectors[class_index])

        self.result_label.config(text=f"Клас {class_index + 1}: {len(file_paths)} образів завантажено.")
        self.display_image(self.class_images[class_index][0])

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        return resized.flatten() / 255.0

    def compute_centroid(self, feature_vectors):
        return np.mean(feature_vectors, axis=0)

    def load_unknown_image(self):
        unknown_path = filedialog.askopenfilename(title="Виберіть невідомий образ", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not unknown_path:
            return
        
        unknown_image = cv2.imread(unknown_path)
        unknown_features = self.extract_features(unknown_image)
        distances = [self.manhattan_distance(unknown_features, centroid) for centroid in self.centroids]

        self.display_image(unknown_image)
        self.display_comparison(distances)

    def manhattan_distance(self, vec1, vec2):
        return np.sum(np.abs(vec1 - vec2))

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.image_label.config(image=image)
        self.image_label.image = image

    def display_comparison(self, distances):
        min_distance = min(distances)
        class_index = np.argmin(distances)

        comparison_text = "Відстані до класів:\n"
        for i, distance in enumerate(distances):
            comparison_text += f"Клас {i + 1}: {distance:.4f}\n"

        comparison_text += f"\nНевідомий образ віднесено до класу {class_index + 1} (мінімальна відстань: {min_distance:.4f})."

        self.result_label.config(text=comparison_text)

    def view_class_images(self, class_index):
        images = self.class_images[class_index]
        if not images:
            messagebox.showinfo("Інформація", f"Клас {class_index + 1} не має завантажених образів.")
            return

        view_window = Toplevel(self.master)
        view_window.title(f"Образи класу {class_index + 1}")
        view_window.geometry("1000x400")
        view_window.configure(bg="#0850bc")

        frame = Frame(view_window, bg="#0850bc")
        frame.pack(pady=10)

        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            label = Label(frame, image=img, bg="#0850bc")
            label.image = img
            label.pack(side=LEFT, padx=5)

if __name__ == "__main__":
    root = Tk()
    app = RecognitionSystem(root)
    root.mainloop()
