import torch
import os
from segmentation import process_image, get_characters_images, reshape_character, CHARS
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# Ładowanie modelu
model_path = "../models/char_recognition/weights/best_weights_only.pt"
model = efficientnet_b1()
features_in = model.classifier[1].in_features
model.classifier = torch.nn.Linear(features_in, len(CHARS))
model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
model.eval()
preprocess = EfficientNet_B1_Weights.DEFAULT.transforms()

# W wyróżniku pojazdu (drugiej części tablicy rejestracyjnej) mie można stosować
# liter B, D, I, O, Z, zbyt podobnych do cyfr 8, 0, 1, 0, 2., dlatego gdy model
# przewidzi taka literę w drugiej części rejestracji, wybieramy jego najpewniejszą
# predykcję nie będącą zakazanym znakiem

# Definicja zakazanych liter
forbidden_chars = {'B', 'D', 'I', 'O', 'Z'}

# Funkcja do rozpoznawania znaków
def recognize_characters(char_images):
    recognized_text = ''
    for char_img in char_images:
        # Przygotowanie obrazu do predykcji
        char_img_resized = reshape_character(char_img)
        char_img_tensor = torch.tensor(char_img_resized).float()
        char_img_tensor = char_img_tensor.repeat(3, 1, 1).unsqueeze(0)
        char_img_tensor = preprocess(char_img_tensor)
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
    test_folder = "."
    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        if image_path.endswith('.png') or image_path.endswith('.jpg'):
            result_text = process_license_plate(image_path)
            print(f"Wynik dla {image_name}: {result_text}")

            # with open(os.path.join(result_folder, f"{image_name}_result.txt"), 'w') as result_file:
            #     result_file.write(result_text)
