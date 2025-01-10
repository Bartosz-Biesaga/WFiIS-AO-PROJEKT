import torch
import cv2
import os
from segmentation import process_image, get_characters_images, reshape_character, CHARS

# Ładowanie modelu
model_path = "models/char_recognition/weights/best_weights_only.pt"
model = torch.load(model_path)
model.eval()

# [ Jedna, dwie lub trzy pierwsze litery tworzą wyróżnik miejsca. Pierwsza litera oznacza województwo. 
# Jedna lub dwie następne – powiat albo miasto na prawach powiatu, albo dzielnicę Warszawy, ale 
# istnieją odstępstwa od tej reguły. Po wyróżniku powiatu (a w przypadku tablic samochodowych 
# zmniejszonych – województwa) następuje wyróżnik pojazdu. Nie można w nim jednak stosować 
# liter B, D, I, O, Z[3],zbyt podobnych do cyfr 8, 0, 1, 0, 2. ]

# Definicja zakazanych liter
forbidden_chars = {'B', 'D', 'I', 'O', 'Z'}

# Funkcja do rozpoznawania znaków
def recognize_characters(char_images):
    recognized_text = ''
    for char_img in char_images:
        # Przygotowanie obrazu do predykcji
        char_img_resized = reshape_character(char_img)
        char_img_tensor = torch.tensor(char_img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0  # Skalowanie do zakresu 0-1

        # Przewidywanie
        with torch.no_grad():
            output = model(char_img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_char = list(CHARS.keys())[predicted.item()]
            
            # Sprawdzanie, czy litera należy do zakazanych
            if predicted_char in forbidden_chars:
                # Jeśli tak, przewidujemy kolejne możliwe litery (np. B -> 8) i tak dalej..
                alternative_predictions = []
                for idx in range(len(output[0])):
                    if idx != predicted.item():
                        alternative_predictions.append((list(CHARS.keys())[idx], output[0][idx].item()))

                # Zostawiamy alternatywne litery posortowane według pewności przewidywań
                alternative_predictions = sorted(alternative_predictions, key=lambda x: x[1], reverse=True)
                recognized_text += f"[{predicted_char} -> {alternative_predictions[0][0]}]"
            else:
                recognized_text += predicted_char
    return recognized_text

def process_license_plate(image_path):
    # Przetwarzanie obrazu tablicy rejestracyjnej
    processed_image = process_image(image_path=image_path)

    # Segmentacja znaków
    char_images, _ = get_characters_images(processed_image)

    # Rozpoznawanie znaków
    result_text = recognize_characters(char_images)
    
    # Jeśli jest spacja, sprawdzamy drugą część
    parts = result_text.split(' ')
    if len(parts) > 1:
        second_part = parts[1]
        for i, char in enumerate(second_part):
            if char in forbidden_chars:
                result_text = result_text.replace(char, f"[{char} -> alternatives]")
                break

    return result_text

if __name__ == "__main__":
    test_folder = "test_data"
    result_folder = "result"

    os.makedirs(result_folder, exist_ok=True)

    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        if image_path.endswith('.png') or image_path.endswith('.jpg'):
            result_text = process_license_plate(image_path)
            print(f"Wynik dla {image_name}: {result_text}")

            with open(os.path.join(result_folder, f"{image_name}_result.txt"), 'w') as result_file:
                result_file.write(result_text)
