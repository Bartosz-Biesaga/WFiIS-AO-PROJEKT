import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from YOLO_utils import crop_boxes_from_image
from recognition import load_model, process_license_plate

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detekcja aut i odczytywanie rejestracji")
        self.root.geometry("1200x600")
        self.root.configure(bg="#f0f0f0")

        button_style = {
            "font": ("Arial", 12),
            "bg": "#5cb85c",
            "fg": "white",
            "relief": "flat",
            "activebackground": "#4cae4c",
        }

        # Inicjalizacja YOLO
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(dir_path, "..", "models", "YOLO", "weights", "best.pt")
        self.yolo = YOLO(weights_path)

        # Inicjalizacja modelu rozpoznawania tablic rejestracyjnych
        self.recognition_model, self.preprocess = load_model()

        # Ścieżki
        self.results_dir = os.path.join(dir_path, "..", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Przechowywanie ścieżki do obrazu użytkownika i wyników detekcji
        self.image_path = None
        self.detections = []
        self.current_car_index = 0

        # Elementy GUI
        self.frame_images = tk.Frame(self.root, bg="#f0f0f0")
        self.frame_images.pack(pady=10, fill=tk.BOTH, expand=True)

        # Etykiety dla obrazów
        self.original_label = tk.Label(self.frame_images, bg="#f0f0f0", text="Oryginalny obraz")
        self.original_label.pack(side="left", padx=10, pady=10, expand=True)

        self.annotated_label = tk.Label(self.frame_images, bg="#f0f0f0", text="Obraz z oznaczeniami")
        self.annotated_label.pack(side="left", padx=10, pady=10, expand=True)

        self.cropped_label = tk.Label(self.frame_images, bg="#f0f0f0", text="Wycinek auta z tablicą")
        self.cropped_label.pack(side="left", padx=10, pady=10, expand=True)

        # Przyciski
        self.select_button = tk.Button(
            self.root, text="Zaimportuj Zdjęcie", command=self.select_image, **button_style
        )
        self.select_button.pack(pady=5)

        self.detect_button = tk.Button(
            self.root, text="Wykryj auta i tablice", command=self.detect_objects, state=tk.DISABLED, **button_style
        )
        self.detect_button.pack(pady=5)

        self.process_button = tk.Button(
            self.root, text="Rozpoznaj tablicę", command=self.process_plate, state=tk.DISABLED, **button_style
        )
        self.process_button.pack(pady=5)

        self.save_button = tk.Button(
            self.root, text="Zapisz Wyniki", command=self.save_results, state=tk.DISABLED, **button_style
        )
        self.save_button.pack(pady=5)

        # Przyciski nawigacji dla wykrytych aut
        self.nav_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.nav_frame.pack(pady=10)

        self.prev_button = tk.Button(
            self.nav_frame, text="< Poprzednie", command=self.prev_car, state=tk.DISABLED, **button_style
        )
        self.prev_button.pack(side="left", padx=5)

        self.next_button = tk.Button(
            self.nav_frame, text="Następne >", command=self.next_car, state=tk.DISABLED, **button_style
        )
        self.next_button.pack(side="left", padx=5)

        # Pole tekstowe do wyświetlania rozpoznanego tekstu tablicy
        self.plate_text_label = tk.Label(self.root, text="Rozpoznany tekst tablicy:", font=("Arial", 12), bg="#f0f0f0")
        self.plate_text_label.pack(pady=5)

        self.plate_text_box = tk.Text(self.root, height=2, width=50, font=("Arial", 12), state=tk.DISABLED)
        self.plate_text_box.pack(pady=5)

    def select_image(self):
        """Pozwól użytkownikowi wybrać plik obrazu i zresetuj GUI."""
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz", filetypes=[("Pliki graficzne", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.detections = []
            self.current_car_index = 0

            # Zresetuj GUI
            self.reset_display()

            # Wyświetl wybrany obraz
            pil_img = Image.open(file_path)
            pil_img.thumbnail((400, 300))
            self.original_photo = ImageTk.PhotoImage(pil_img)
            self.original_label.config(image=self.original_photo, text="")
            self.original_label.image = self.original_photo

            # Włącz przycisk detekcji
            self.detect_button.config(state=tk.NORMAL)

    def reset_display(self):
        """Zresetuj wszystkie wyświetlane obrazy i tekst."""
        self.annotated_label.config(image="", text="Obraz z oznaczeniami")
        self.cropped_label.config(image="", text="Wycinek auta z tablicą")
        self.plate_text_box.config(state=tk.NORMAL)
        self.plate_text_box.delete("1.0", tk.END)
        self.plate_text_box.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)

    def detect_objects(self):
        """Wykryj auta i tablice rejestracyjne na wybranym obrazie."""
        if not self.image_path:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wybierz obraz.")
            return

        cv_img = cv2.imread(self.image_path)
        if cv_img is None:
            messagebox.showerror("Błąd", "Nie można odczytać obrazu.")
            return

        # Wykrywanie obiektów za pomocą YOLO
        try:
            self.detections = crop_boxes_from_image(self.yolo, cv_img, save_prediction=True)
            self.detections = [d for d in self.detections if d[1]]  # Filtruj auta bez tablic

            if not self.detections:
                messagebox.showinfo("Informacja", "Nie wykryto aut z tablicami rejestracyjnymi.")
                return

            # Wyświetl obraz z oznaczeniami
            annotated_path = os.path.join(self.results_dir, "prediction.jpg")
            if os.path.exists(annotated_path):
                pil_annotated = Image.open(annotated_path)
                pil_annotated.thumbnail((400, 300))
                self.annotated_photo = ImageTk.PhotoImage(pil_annotated)
                self.annotated_label.config(image=self.annotated_photo, text="")
                self.annotated_label.image = self.annotated_photo

            # Pokaż pierwsze auto
            self.current_car_index = 0
            self.show_current_car()

            # Włącz przyciski nawigacji i przetwarzania
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Błąd", f"Detekcja nie powiodła się: {e}")

    def show_current_car(self):
        """Wyświetl aktualnie wybrane auto i jego tablicę."""
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        car, plates = self.detections[self.current_car_index]
        if car is not None:
            car_img = Image.fromarray(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
            car_img.thumbnail((400, 300))
            self.cropped_photo = ImageTk.PhotoImage(car_img)
            self.cropped_label.config(image=self.cropped_photo, text="")
            self.cropped_label.image = self.cropped_photo

    def prev_car(self):
        """Pokaż poprzednie auto na liście detekcji."""
        if self.current_car_index > 0:
            self.current_car_index -= 1
            self.show_current_car()

    def next_car(self):
        """Pokaż następne auto na liście detekcji."""
        if self.current_car_index < len(self.detections) - 1:
            self.current_car_index += 1
            self.show_current_car()

    def process_plate(self):
        """Przetwarzaj tablicę rejestracyjną aktualnie wybranego auta."""
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        car, plates = self.detections[self.current_car_index]
        if plates:
            plate = plates[0]  # Przetwarzaj pierwszą wykrytą tablicę
            try:
                text = process_license_plate(
                    image=plate, model=self.recognition_model, preprocess=self.preprocess
                )
                self.plate_text_box.config(state=tk.NORMAL)
                self.plate_text_box.delete("1.0", tk.END)
                self.plate_text_box.insert(tk.END, text)
                self.plate_text_box.config(state=tk.DISABLED)

                # Włącz przycisk zapisu
                self.save_button.config(state=tk.NORMAL)

            except Exception as e:
                messagebox.showerror("Błąd", f"Przetwarzanie tablicy rejestracyjnej nie powiodło się: {e}")

    def save_results(self):
        """Zapisz aktualnie wyświetlane auto i tekst tablicy do katalogu wyników."""
        if not self.detections or self.current_car_index >= len(self.detections):
            return

        car, plates = self.detections[self.current_car_index]
        if car is not None:
            car_path = os.path.join(self.results_dir, f"car_{self.current_car_index}.jpg")
            cv2.imwrite(car_path, cv2.cvtColor(car, cv2.COLOR_RGB2BGR))

        plate_text = self.plate_text_box.get("1.0", tk.END).strip()
        if plate_text:
            text_path = os.path.join(self.results_dir, f"plate_{self.current_car_index}.txt")
            with open(text_path, "w") as f:
                f.write(plate_text)

        messagebox.showinfo("Zapisano", "Wyniki zostały zapisane!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
