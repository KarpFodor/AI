import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recognition of handwritten numbers")
        
        # Зберігання даних
        self.current_points = []
        self.freeman_code = []
        self.digit_classes = {str(i): [] for i in range(1, 10)}
        self.load_digit_classes()
        
        self.min_line_length = 1.0

        self.setup_gui()
        
    def setup_gui(self):
        # Створення головного контейнера
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Створення двох графіків
        self.fig = Figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
        # Налаштування графіків
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.grid(True)
        
        self.ax1.set_title("Еталонна цифра")
        self.ax2.set_title("Тестова цифра")
        
        # Кнопки керування
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="Порівняйти", command=self.compare_digits).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Показати все", command=self.show_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистити", command=self.clear_plot).pack(side=tk.LEFT, padx=5)

        # Line length slider
        line_length_frame = ttk.Frame(main_frame)
        line_length_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Label(line_length_frame, text="Довжина лінії").pack(side=tk.LEFT, padx=5)
        self.line_length_slider = ttk.Scale(line_length_frame, from_=0.1, to=5.0, 
                                            orient=tk.HORIZONTAL, command=self.update_line_length)
        self.line_length_slider.set(self.min_line_length)
        self.line_length_slider.pack(side=tk.LEFT)

        # Кнопки цифр
        digits_frame = ttk.Frame(main_frame)
        digits_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        for i in range(1, 10):
            ttk.Button(digits_frame, text=str(i), 
                      command=lambda x=i: self.save_to_class(str(x))).pack(side=tk.LEFT, padx=2)
        
        # Прив'язка подій миші
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.drawing = False
        self.current_ax = None
        
    def update_line_length(self, value):
        self.min_line_length = float(value)

    def on_click(self, event):
        if event.inaxes in [self.ax1, self.ax2]:
            self.drawing = True
            self.current_ax = event.inaxes
            self.current_points = [(event.xdata, event.ydata)]
            plot_name = "еталонному" if self.current_ax == self.ax1 else "тестовому"
            print(f"\nПочаток малювання на {plot_name} графіку")
            
    def on_motion(self, event):
        if self.drawing and event.inaxes == self.current_ax:
            last_point = self.current_points[-1]
            distance = np.hypot(event.xdata - last_point[0], event.ydata - last_point[1])

            if distance >= self.min_line_length:
                self.current_points.append((event.xdata, event.ydata))
                dx = event.xdata - last_point[0]
                dy = event.ydata - last_point[1]
                angle = np.arctan2(dy, dx)
                freeman_direction = int(((angle + np.pi) * 4 / np.pi + 0.5) % 8)
                direction_arrow = self.get_direction_arrow(freeman_direction)

                self.current_ax.plot([last_point[0], event.xdata],
                                     [last_point[1], event.ydata], 'r-')
                self.current_ax.annotate(f"{freeman_direction} {direction_arrow}", 
                                         xy=(event.xdata, event.ydata), 
                                         textcoords="offset points", 
                                         xytext=(5, 5), 
                                         ha='center', fontsize=8, color='blue')
                
                self.freeman_code.append(freeman_direction)
                self.canvas.draw()
            
    def get_direction_arrow(self, direction):
        arrows = {
            4: "→", 5: "↗", 6: "↑", 7: "↖", 0: "←", 1: "↙", 2: "↓", 3: "↘"
        }
        return arrows.get(direction, "")
        
    def on_release(self, event):
        if self.drawing:
            self.drawing = False
            self.optimize_freeman_code()
            plot_name = "еталонному" if self.current_ax == self.ax1 else "тестовому"
            print(f"Завершено малювання на {plot_name} графіку")
            print("Оптимізований код Фрімена:", self.freeman_code)
            print(f"Довжина коду: {len(self.freeman_code)}")
            self.print_freeman_directions(self.freeman_code)

    def print_freeman_directions(self, code):
        """Виведення напрямків руху для кожного вектора коду Фрімена"""
        directions = {
            4: "→ (вправо)",
            5: "↗ (вправо-вгору)",
            6: "↑ (вгору)",
            7: "↖ (вліво-вгору)",
            0: "← (вліво)",
            1: "↙ (вліво-вниз)",
            2: "↓ (вниз)",
            3: "↘ (вправо-вниз)"
        }
        print("\nНапрямки руху:")
        for i, direction in enumerate(code):
            print(f"Вектор {i+1}: {direction} - {directions[direction]}")
            
    def calculate_freeman_code(self, points):
        code = []
        for i in range(len(points)-1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            angle = np.arctan2(dy, dx)
            # Конвертація кута в код Фрімена (0-7)
            freeman_direction = int(((angle + np.pi) * 4 / np.pi + 0.5) % 8)
            code.append(freeman_direction)
        return code
    
    def optimize_freeman_code(self):
        # Видалення послідовних повторень
        if not self.freeman_code:
            return
        print("\nОптимізація коду Фрімена:")
        print("До оптимізації:", self.freeman_code)
        optimized = [self.freeman_code[0]]
        for code in self.freeman_code[1:]:
            if code != optimized[-1]:
                optimized.append(code)
        self.freeman_code = optimized
        print("Після оптимізації:", self.freeman_code)
        print(f"Кількість векторів зменшено з {len(self.freeman_code)} до {len(optimized)}")
        
    def compare_digits(self):
        if not self.freeman_code:
            messagebox.showwarning("Попередження", "Спочатку намалюйте цифру!")
            return
            
        print("\nПорівняння цифр:")
        print("Тестовий код:", self.freeman_code)
        
        similarity_scores = {}
        test_code = self.freeman_code
        
        for digit, codes in self.digit_classes.items():
            if codes:  # Якщо є збережені коди для цього класу
                print(f"\nПорівняння з класом {digit}:")
                # Обчислюємо подібність з усіма зразками класу
                similarities = []
                for i, stored_code in enumerate(codes):
                    similarity = self.calculate_similarity(test_code, stored_code)
                    similarities.append(similarity)
                    print(f"Зразок {i+1}: {stored_code} (схожість: {similarity:.2f}%)")
                max_similarity = max(similarities)
                similarity_scores[digit] = max_similarity
                print(f"Максимальна схожість для класу {digit}: {max_similarity:.2f}%")
        
        if similarity_scores:
            # Сортування результатів
            sorted_scores = sorted(similarity_scores.items(), 
                                key=lambda x: x[1], reverse=True)
            
            # Виведення результатів
            result = "Similarity Scores:\n\n"
            print("\nФінальні результати:")
            for digit, score in sorted_scores:
                result += f"Class {digit}: {score:.2f}%\n"
                print(f"Клас {digit}: {score:.2f}%")
            
            messagebox.showinfo("Результати порівняння", result)
        else:
            print("\nНемає збережених зразків для порівняння")
            messagebox.showinfo("Інформація", "Немає збережених зразків для порівняння")
    
    def calculate_similarity(self, code1, code2):
        # Простий алгоритм порівняння послідовностей кодів Фрімена
        if not code1 or not code2:
            return 0
            
        len1, len2 = len(code1), len(code2)
        max_len = max(len1, len2)
        
        # Нормалізація довжин кодів
        code1_normalized = code1 * (max_len // len1 + 1)
        code2_normalized = code2 * (max_len // len2 + 1)
        
        # Обрізання до однакової довжини
        code1_normalized = code1_normalized[:max_len]
        code2_normalized = code2_normalized[:max_len]
        
        # Підрахунок співпадінь
        matches = sum(1 for i in range(max_len) 
                     if code1_normalized[i] == code2_normalized[i])
        
        return (matches / max_len) * 100
    
    def save_to_class(self, digit_class):
        if self.freeman_code:
            self.digit_classes[digit_class].append(self.freeman_code)
            self.save_digit_classes()
            print(f"\nЗбереження у клас {digit_class}:")
            print(f"Код Фрімена: {self.freeman_code}")
            print(f"Всього зразків у класі {digit_class}: {len(self.digit_classes[digit_class])}")
            messagebox.showinfo("Збереження", 
                              f"Цифру збережено в клас {digit_class}")
        else:
            messagebox.showwarning("Попередження", 
                                 "Спочатку намалюйте цифру!")
    
    def show_all(self):
        # Показати всі збережені зразки
        print("\nЗбережені зразки:")
        info = "Збережені зразки:\n\n"
        for digit, codes in self.digit_classes.items():
            info += f"Клас {digit}: {len(codes)} зразків\n"
            print(f"\nКлас {digit} ({len(codes)} зразків):")
            for i, code in enumerate(codes):
                print(f"Зразок {i+1}: {code}")
        messagebox.showinfo("Інформація", info)
    
    def clear_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.grid(True)
        self.ax1.set_title("Еталонна цифра")
        self.ax2.set_title("Тестова цифра")
        self.canvas.draw()
        self.current_points = []
        self.freeman_code = []
        print("\nГрафіки очищено")
    
    def save_digit_classes(self):
        # Зберігання даних у файл
        with open('classes.json', 'w') as f:
            json.dump(self.digit_classes, f)
        print("\nДані збережено у файл")
    
    def load_digit_classes(self):
        # Завантаження даних з файлу
        try:
            with open('classes.json', 'r') as f:
                self.digit_classes = json.load(f)
            print("\nЗавантажено збережені дані")
            print("Кількість зразків у кожному класі:")
            for digit, codes in self.digit_classes.items():
                print(f"Клас {digit}: {len(codes)} зразків")
        except FileNotFoundError:
            print("\nФайл зі збереженими даними не знайдено")
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()