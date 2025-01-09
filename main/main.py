import os
from segmentation import process_image, get_characters_images

# Ścieżki do folderów z danymi i wynikami
input_folder = "test_data"
output_folder = "results"

# Ustawienia
blur_ksize = 5  # Rozmiar filtra rozmycia
debug = True  # Czy zapisywać debugowe obrazy z etapów przetwarzania

# Tworzenie folderu na wyniki, jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# Iteracja po obrazach w folderze wejściowym
for image_name in os.listdir(input_folder):
    # Sprawdzenie, czy plik to obraz
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, image_name)
        print(f"Przetwarzanie obrazu: {image_name}")

        # Ścieżka do folderu debugowego dla tego obrazu
        debug_folder = os.path.join(output_folder, f"debug_{image_name.split('.')[0]}") if debug else None

        # Przetwarzanie obrazu (usuwanie euroband i binaryzacja)
        processed_img = process_image(image_path, debug_folder=debug_folder, blur_ksize=blur_ksize)

        # Wydzielanie znaków z przetworzonego obrazu
        characters = get_characters_images(processed_img, debug_folder=debug_folder)

        # Zapis wyciętych znaków do plików
        char_output_folder = os.path.join(output_folder, f"chars_{image_name.split('.')[0]}")
        os.makedirs(char_output_folder, exist_ok=True)

        for i, char_img in enumerate(characters):
            char_path = os.path.join(char_output_folder, f"char_{i + 1}.png")
            char_img.save(char_path)  # Zapis jako plik PNG
            print(f"  Zapisano znak {i + 1}: {char_path}")

print("Przetwarzanie zakończone.")
