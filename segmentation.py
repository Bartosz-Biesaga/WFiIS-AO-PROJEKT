import os
import cv2
import numpy as np

def process_image(image_path, debug=False, aspect_ratio_range=(2, 5), min_height=20, blur_ksize=(5, 5)):
    debug_folder = "debug"
    if debug:
        os.makedirs(debug_folder, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        return "Błąd: Nie można wczytać obrazu", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "gray.jpg"), gray)

    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "blurred.jpg"), blurred)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "thresh.jpg"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    white_background = np.ones_like(license_plate) * 255
    _, thresh_plate = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_license_plate = cv2.bitwise_and(white_background, white_background, mask=thresh_plate)
    final_license_plate = cv2.add(masked_license_plate, thresh_plate)
    
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "license_plate_white_background.jpg"), final_license_plate)

    return "Tablica znaleziona", final_license_plate


def save_characters_images(license_plate_image, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    inverted_plate = cv2.bitwise_not(license_plate_image)
    
    contours, _ = cv2.findContours(inverted_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    characters = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        char_image = license_plate_image[y:y+h, x:x+w]
        char_path = os.path.join(output_folder, f"char_{i}.jpg")
        cv2.imwrite(char_path, char_image)
        characters.append(char_image)

    return characters, len(contours)


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
