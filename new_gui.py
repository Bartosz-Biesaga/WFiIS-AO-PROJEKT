import tkinter as tk
from tkinter import filedialog, messagebox
import os

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

from yolo_detection import crop_boxes_from_image
from recognition import load_model, process_license_plate

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

        # Initialize YOLO
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(dir_path, "best.pt")
        self.yolo = YOLO(weights_path)

        # Initialize the license plate recognition model
        self.recognition_model, self.preprocess = load_model()

        # Paths
        self.results_dir = os.path.join(dir_path, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Store the user's image path
        self.image_path = None

        # GUI Elements

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

        # Textbox to display recognized plate text
        self.plate_text_label = tk.Label(self.root, text="Rozpoznany tekst tablicy:", font=("Arial", 12), bg="#f0f0f0")
        self.plate_text_label.pack(pady=5)

        self.plate_text_box = tk.Text(self.root, height=2, width=50, font=("Arial", 12), state=tk.DISABLED)
        self.plate_text_box.pack(pady=5)

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
                pil_img.thumbnail((400, 300))
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
          # Detect cars and license plates
          results = self.yolo(cv_img)

          # Save the annotated image
          annotated_img = results[0].plot()
          annotated_dest = os.path.join(self.results_dir, "annotated_image.jpg")
          cv2.imwrite(annotated_dest, annotated_img)

          # Display the annotated image
          if os.path.exists(annotated_dest):
              pil_annotated = Image.open(annotated_dest)
              pil_annotated.thumbnail((400, 300))
              self.annotated_photo = ImageTk.PhotoImage(pil_annotated)
              self.annotated_label.config(image=self.annotated_photo, text="")
              self.annotated_label.image = self.annotated_photo
          else:
              self.annotated_label.config(text="No annotated image found")

          # Process license plates and display recognized text
          recognized_texts = []
          for result in results:
              if not result.boxes or len(result.boxes.xyxy) == 0:
                  print("No bounding boxes found for this detection result.")
                  continue

              boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
              for box in boxes:
                  x1, y1, x2, y2 = map(int, box[:4])
                  # Validate bounding box dimensions
                  if x2 <= x1 or y2 <= y1:
                      print(f"Invalid bounding box: {box}")
                      continue

                  cropped_plate = cv_img[y1:y2, x1:x2]
                  if cropped_plate.size == 0:
                      print("Cropped plate region is empty.")
                      continue

                  try:
                      text = process_license_plate(
                          image=cropped_plate,
                          model=self.recognition_model,
                          preprocess=self.preprocess
                      )
                      if text.strip():  # Ignore empty or invalid results
                          recognized_texts.append(text)
                  except Exception as e:
                      print(f"Error processing license plate: {e}")
                      recognized_texts.append("[PROCESSING ERROR]")

          if not recognized_texts:
              messagebox.showwarning("Warning", "No license plate text detected.")
          else:
              # Save recognized text to a file in the results directory
              text_file_path = os.path.join(self.results_dir, "recognized_plates.txt")
              with open(text_file_path, "w") as text_file:
                  text_file.write("\n".join(recognized_texts))

              # Display recognized text in the text box
              self.plate_text_box.config(state=tk.NORMAL)
              self.plate_text_box.delete("1.0", tk.END)
              self.plate_text_box.insert(tk.END, "\n".join(recognized_texts))
              self.plate_text_box.config(state=tk.DISABLED)

              messagebox.showinfo("Done", f"Processing complete! Check the results in {text_file_path}.")

      except Exception as e:
          messagebox.showerror("Detection Error", f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
