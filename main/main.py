import torch
import os
from segmentation import process_image, get_characters_images, reshape_character, CHARS
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# Definicja zakazanych liter
forbidden_chars = {'B', 'D', 'I', 'O', 'Z'}

# Inicjalizacja ścieżek
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "..", "models", "char_recognition", "weights", "best_weights_only.pt")

# Funkcja do ładowania modelu
def load_model(model_path):
    try:
        model = efficientnet_b1()
        features_in = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(features_in, len(CHARS))
        model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
        model.eval()
        preprocess = EfficientNet_B1_Weights.DEFAULT.transforms()
        return model, preprocess
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

# Funkcja do rozpoznawania znaków
def recognize_characters(model, preprocess, char_images):
    recognized_text = ''
    for char_img in char_images:
        try:
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
                recognized_text += predicted_char
        except Exception as e:
            recognized_text += "[ERROR]"
            print(f"Error processing character: {e}")
    return recognized_text

# Funkcja przetwarzania tablicy rejestracyjnej
def process_license_plate(image_path):
    try:
        # Przetwarzanie obrazu tablicy rejestracyjnej
        processed_image = process_image(image_path=image_path)

        # Segmentacja znaków
        char_images, _ = get_characters_images(processed_image)

        # Rozpoznawanie znaków
        recognized_text = recognize_characters(model, preprocess, char_images)

        # Dodanie spacji, jeśli jej brak
        if ' ' not in recognized_text and len(recognized_text) > 3:
            recognized_text = recognized_text[:3] + ' ' + recognized_text[3:]

        # Obsługa zakazanych znaków w drugiej części
        parts = recognized_text.split(' ')
        if len(parts) > 1:
            first_part, second_part = parts[0], parts[1]
            corrected_second_part = ''
            for char in second_part:
                if char in forbidden_chars:
                    # Jeśli znak jest zakazany, znajdujemy alternatywę w dozwolonych literach
                    alternative_predictions = []
                    for idx in range(len(output[0])):
                        predicted_char = list(CHARS.keys())[idx]
                        if predicted_char not in forbidden_chars:  # Tylko dozwolone litery
                            alternative_predictions.append((predicted_char, output[0][idx].item()))
                    # Posortowanie alternatyw według pewności przewidywań
                    alternative_predictions = sorted(alternative_predictions, key=lambda x: x[1], reverse=True)
                    corrected_second_part += alternative_predictions[0][0]  # Zamiana na najpewniejszą alternatywę
                else:
                    corrected_second_part += char
            recognized_text = f"{first_part} {corrected_second_part}"

        return recognized_text
    except Exception as e:
        print(f"Error processing license plate: {e}")
        return "[PROCESSING ERROR]"

if __name__ == "__main__":
    # Ładowanie modelu
    model, preprocess = load_model(model_path)

    # Przetwarzanie obrazów w folderze
    test_folder = "."
    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        if image_path.endswith('.png') or image_path.endswith('.jpg'):
            result_text = process_license_plate(image_path)
            print(f"Wynik dla {image_name}: {result_text}")
