import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie Tablic Rejestracyjnych")
        self.root.geometry("854x480")
        self.root.configure(bg="#f0f0f0")

        # Styl dla przycisków
        button_style = {
            "font": ("Arial", 12),
            "bg": "#5cb85c",
            "fg": "white",
            "relief": "flat",
            "activebackground": "#4cae4c",
        }

        # Przycisk do załadowania obrazu
        self.upload_button = tk.Button(
            root,
            text="Załaduj obraz",
            command=self.upload_image,
            **button_style
        )
        self.upload_button.pack(pady=20)

        
        self.image_label = tk.Label(root, bg="#f0f0f0")
        self.image_label.pack()

        # Przycisk do rozpoznawania tablic rejestracyjnych
        self.recognize_button = tk.Button(
            root,
            text="Rozpoznaj tablicę rejestracyjną",
            command=self.recognize_plate,
            **button_style
        )
        self.recognize_button.pack(pady=20)

    def upload_image(self):
        # Funkcja obsługująca załadowanie obrazu
        file_path = filedialog.askopenfilename(
            filetypes=[("Pliki graficzne", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )

        if file_path:
            try:
                image = Image.open(file_path)
                image.thumbnail((400, 300))  
                self.photo = ImageTk.PhotoImage(image)

                self.image_label.configure(image=self.photo)
                self.image_label.image = self.photo

                messagebox.showinfo("Sukces", "Obraz załadowany pomyślnie!")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można otworzyć obrazu: {e}")

    def recognize_plate(self):
        #do zrobienia
        messagebox.showinfo("Funkcja", "Funkcja rozpoznawania tablic wkrótce dostępna!")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
