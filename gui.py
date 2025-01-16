import tkinter as tk
from tkinter import filedialog, messagebox
import os

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from yolo_detection import crop_boxes_from_image

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detection GUI")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")

        # Optional style for buttons
        button_style = {
            "font": ("Arial", 12),
            "bg": "#5cb85c",
            "fg": "white",
            "relief": "flat",
            "activebackground": "#4cae4c",
        }

        # 1. Initialize YOLO here
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(dir_path, "..", "best.pt")
        self.yolo = YOLO(weights_path)

        # Paths
        self.results_dir = os.path.join(dir_path, "..", "results")
        os.makedirs(self.results_dir, exist_ok=True)  # ensure results folder exists

        # We'll store the user's image path
        self.image_path = None

        # 2. Create GUI elements

        # Frame for original and processed images side-by-side
        self.frame_images = tk.Frame(self.root, bg="#f0f0f0")
        self.frame_images.pack(pady=10)

        # Left label: original image
        self.original_label = tk.Label(self.frame_images, bg="#f0f0f0", text="Original Image")
        self.original_label.pack(side="left", padx=20)

        # Right label: annotated image
        self.annotated_label = tk.Label(self.frame_images, bg="#f0f0f0", text="Annotated Image")
        self.annotated_label.pack(side="left", padx=20)

        # Button: Select Image
        self.select_button = tk.Button(
            self.root,
            text="Zaimportuj Zdjęcie",
            command=self.select_image,
            **button_style
        )
        self.select_button.pack(pady=5)

        # Button: Process with YOLO
        self.process_button = tk.Button(
            self.root,
            text="Znajdź tablicę rejestracyjną",
            command=self.process_image,
            state=tk.DISABLED,  # Disabled until user selects an image
            **button_style
        )
        self.process_button.pack(pady=5)

    def select_image(self):
        """Let the user pick an image file, then display it in the 'original_label'."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            self.image_path = file_path
            try:
                # Load with PIL to show in the GUI
                pil_img = Image.open(file_path)
                pil_img.thumbnail((400, 300))  # Make sure it fits the GUI
                self.original_photo = ImageTk.PhotoImage(pil_img)

                # Show in the original_label
                self.original_label.config(image=self.original_photo, text="")
                self.original_label.image = self.original_photo

                # Enable the "Process with YOLO" button
                self.process_button.config(state=tk.NORMAL)

                messagebox.showinfo("Image Selected", f"You chose: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot open image:\n{e}")

    def process_image(self):
        """Runs YOLO detection, saves results to disk, and displays the annotated image."""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Load image via OpenCV (BGR)
        cv_img = cv2.imread(self.image_path)
        if cv_img is None:
            messagebox.showerror("Error", f"Could not read image: {self.image_path}")
            return

        try:
            
            pairs = crop_boxes_from_image(
                self.yolo,
                cv_img,
                save_prediction=True
            )

            
            annotated_src = "prediction.jpg"  
            annotated_dest = os.path.join(self.results_dir, "prediction.jpg")

            
            if os.path.exists(annotated_src):
                os.replace(annotated_src, annotated_dest)

            # 4. Display the annotated image in the 'annotated_label'
            if os.path.exists(annotated_dest):
                pil_annotated = Image.open(annotated_dest)
                pil_annotated.thumbnail((400, 300))
                self.annotated_photo = ImageTk.PhotoImage(pil_annotated)
                self.annotated_label.config(image=self.annotated_photo, text="")
                self.annotated_label.image = self.annotated_photo
            else:
                self.annotated_label.config(text="No annotated image found")

            
            for i, (car_img, plates_list) in enumerate(pairs):
                if car_img is not None:
                    car_filename = os.path.join(self.results_dir, f"car_{i}.jpg")
                    cv2.imwrite(car_filename, car_img)
                    for j, plate_img in enumerate(plates_list):
                        plate_filename = os.path.join(
                            self.results_dir, f"license_plate_{i}_{j}.jpg"
                        )
                        cv2.imwrite(plate_filename, plate_img)

            messagebox.showinfo("Done", f"Processing complete!\nCheck {self.results_dir} for results.")

        except Exception as e:
            messagebox.showerror("Detection Error", f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
