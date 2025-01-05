import os
import cv2
import numpy as np

def process_image(image_path, debug=False, aspect_ratio_range=(2, 5), min_height=20, blur_ksize=(5, 5)):
    """
    Przetwarza obraz w celu wykrycia i wyizolowania tablicy rejestracyjnej.

    Argumenty:
        image_path (str): Ścieżka do obrazu.
        debug (bool): Zapisz obrazy pomocnicze do debugowania, jeśli True.
        aspect_ratio_range (tuple): Oczekiwany zakres proporcji boków dla tablicy rejestracyjnej.
        min_height (int): Minimalna wysokość tablicy rejestracyjnej w pikselach.
        blur_ksize (tuple): Rozmiar jądra dla rozmycia Gaussa.

    Zwraca:
        tuple: Komunikat statusu oraz przetworzony obraz tablicy rejestracyjnej lub None.
    """
    debug_folder = "debug"
    if debug:
        os.makedirs(debug_folder, exist_ok=True)

    # Wczytanie obrazu
    image = cv2.imread(image_path)
    if image is None:
        return "Błąd: Nie można wczytać obrazu", None

    # Konwersja do odcieni szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "gray.jpg"), gray)

    # Zastosowanie rozmycia Gaussa
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "blurred.jpg"), blurred)

    # Zastosowanie adaptacyjnego progowania
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "thresh.jpg"), thresh)

    # Wyszukiwanie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Wykrywanie tablicy rejestracyjnej
    license_plate = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and h > min_height:
            license_plate = gray[y:y+h, x:x+w]
            if debug:
                plate_debug = image[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(debug_folder, "license_plate.jpg"), plate_debug)
            break

    if license_plate is None:
        return "Tablica nie znaleziona", None

    # Tworzenie białego tła
    white_background = np.ones_like(license_plate) * 255

    # Zastosowanie progowania binaryzującego do uzyskania postaci znaków
    _, thresh_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Maskowanie znaków na białym tle
    masked_license_plate = cv2.bitwise_and(white_background, white_background, mask=thresh_plate)

    # Zachowanie znaków na białym tle
    final_license_plate = cv2.add(masked_license_plate, thresh_plate)
    
    # Zapisanie lub debuggowanie finalnego obrazu
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "license_plate_white_background.jpg"), final_license_plate)

    return "Tablica znaleziona", final_license_plate


def save_characters_images(license_plate_image, output_folder, possible_chars=(7, 8)):
    """
    Dzieli obraz tablicy rejestracyjnej na poszczególne znaki i zapisuje je.

    Argumenty:
        license_plate_image (ndarray): Przetworzony obraz tablicy rejestracyjnej.
        output_folder (str): Katalog, w którym będą zapisywane obrazy znaków.
        possible_chars (tuple): Możliwe liczby znaków do wykrycia.

    Zwraca:
        tuple: Lista ścieżek do zapisanych obrazów znaków oraz liczba wykrytych znaków.
    """
    os.makedirs(output_folder, exist_ok=True)

    h, w = license_plate_image.shape
    best_fit, best_variance = None, float('inf')

    # Określenie najlepszego dopasowania dla segmentacji znaków
    for chars in possible_chars:
        char_width = w // chars
        estimated_width = char_width * chars
        variance = abs(w - estimated_width)
        if variance < best_variance:
            best_variance, best_fit = variance, chars

    char_width = w // best_fit
    characters = []

    # Segmentacja znaków na podstawie najlepszego dopasowania
    for i in range(best_fit):
        char_image = license_plate_image[:, i * char_width:(i + 1) * char_width]
        char_path = os.path.join(output_folder, f"char_{i}.jpg")
        cv2.imwrite(char_path, char_image)
        characters.append(char_path)

    return characters, best_fit


def main():
    dataset_path = '.'
    if not os.path.exists(dataset_path):
        print(f"Folder {dataset_path} nie istnieje!")
        return

    all_characters = []

    for file_name in os.listdir(dataset_path):
        if file_name.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(dataset_path, file_name)
            print(f"Przetwarzanie: {image_path}")

            status, license_plate_image = process_image(image_path, debug=True)
            if status == "Tablica znaleziona" and license_plate_image is not None:
                output_folder = os.path.join("debug", f"char_images_{file_name.split('.')[0]}")
                characters, char_count = save_characters_images(license_plate_image, output_folder)
                all_characters.append({"plik": file_name, "znaki": char_count, "ścieżki": characters})
            else:
                all_characters.append({"plik": file_name, "znaki": 0, "ścieżki": ["Brak tablicy"]})

    for row in all_characters:
        print(f"Plik: {row['plik']}, Znaki: {row['znaki']}, Ścieżki: {row['ścieżki']}")
    return all_characters


if __name__ == "__main__":
    main()
