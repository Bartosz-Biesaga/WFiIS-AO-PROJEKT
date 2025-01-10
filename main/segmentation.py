import os, cv2
import numpy as np
from skimage.filters import threshold_local
import xml.etree.ElementTree as ET

# przetworzenie tablicy rejestracyjnej
def process_image(image_path = None, img = None, debug_folder=None, blur_ksize=(5, 5)):
    # folder na podgląd poszczególnych etapów przetwarzania
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # wczytywanie obrazka tablicy rejestracyjnej ze ścieżki
    if image_path is not None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Błąd: Nie można wczytać obrazu")
    # używanie przekazanego obrazka tablicy rejestracyjnej
    elif img is not None:
        image = img
    else:
        raise ValueError("Nie przekazano obrazu lub sciezki")

    save_count = 0
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_org.png"), image)
        save_count += 1

    # rozmyty obrazek, polepsza efekty binaryzacji
    image = cv2.GaussianBlur(image, blur_ksize, 0)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_blurred.png"), image)
        save_count += 1
    # kanał niebieski i czerwony, używane do usunięcia eurobandu
    # alternatywnie możnaby skorzystać w tym celu ze składowej H modelu HSV
    # (ale nie zostało to przetestowane)
    B = image[:,:,0]
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_blue.png"), B)
        save_count += 1
    R = image[:,:,2]
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_red.png"), R)
        save_count += 1
    Bh, Bw = B.shape[:]
    # wartość bezwględna różnicy między kanałem niebieskim a czerwonym, przy
    # jednorodnie oświetlonych obrazach pozwala na skuteczną lokalizację eurobandu
    abs_diff = np.zeros(B.shape[:], dtype=np.uint8)
    for h in range(Bh):
        for w in range(Bw):
            abs_diff[h][w] = abs(int(B[h][w])-int(R[h][w]))
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_BR_diff.png"), abs_diff)
        save_count += 1
    # maska pozwalająca usunąć euroband
    _, diff_thresh = cv2.threshold(abs_diff, 40, 255, cv2.THRESH_BINARY)
    diff_thresh = cv2.bitwise_not(diff_thresh)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_BR_diff_thresh.png"), diff_thresh)
        save_count += 1

    # kanał value z modelu HSV pozwala na precyzyjne oddzielenie znaków z tablicy
    # rejestracyjnej
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_value.png"), V)
        save_count += 1
    # progowanie adaptacyjne kanału value
    T = threshold_local(V, 61, offset=15, method="gaussian")
    image = (V > T).astype("uint8") * 255
    # negacja obrazu, aby obiekty były białem
    image = cv2.bitwise_not(image)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_thresh.png"), image)
        save_count += 1
    # iloczyn z maską usuwającą euroband
    image = cv2.bitwise_and(image, diff_thresh)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_thresh_bitand.png"), image)
        save_count += 1
    return image

# sprawdza czy dany kontur dotyka krawędzi obrazka
def touches_border(contour, lh, lw):
    x, y, w, h = cv2.boundingRect(contour)
    if x <= 1 or y <= 1 or abs(lw - (x + w)) <= 1 or abs(lh - (y + h)) <= 1:
        return True
    else:
        return False

# funkcja wydzielająca segmenty, w których znajdują się litery
def get_characters_images(license_plate_image, debug_folder = None, min_char_height_factor=0.4,
                    max_char_height_factor=0.9, min_char_aspect_ratio=1., max_char_aspect_ratio=10.,
                    min_char_width_factor = 0.015, max_char_width_factor = 0.18):
    # folder na podgląd poszczególnych etapów przetwarzania
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok = True)

    image = license_plate_image
    save_count = 0

    # znalezienie konturów obiektów z przekazanego obrazka
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # określenie warunków określających czy dany obiekt jest znakiem
    lh, lw = license_plate_image.shape[:2]
    min_char_height = min_char_height_factor * lh
    max_char_height = max_char_height_factor * lh
    min_char_width = min_char_width_factor * lw
    max_char_width = max_char_width_factor * lw
    # maska na znaki, pozwalająca oczyścić obrazek z niepotrzebnych konturów
    chars_mask = np.zeros(image.shape[:], dtype=np.uint8)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        # jesli dany kontur spełnia warunki, to rysujemy jego wypełnienie na masce
        if (min_char_height <= h <= max_char_height and min_char_aspect_ratio <= aspect_ratio <= max_char_aspect_ratio
                and min_char_width <= w <= max_char_width and (touches_border(contour, lh, lw) == False)):
            cv2.drawContours(chars_mask, [contour], 0, (255,), -1)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_chars_mask.png"), chars_mask)
        save_count += 1
    # iloczyn przetworzonej tablicy rejestracyjnej z maską znaków
    chars_image = cv2.bitwise_and(image, chars_mask)
    if debug_folder is not None:
        cv2.imwrite(os.path.join(debug_folder, f"{save_count}_chars_image.png"), chars_image)
        save_count += 1

    # [wycięty znak]
    characters = []
    # [składowa x lewej krawędzi bounding boxa konturu,
    # składowa x prawej krawędzi bounding boxa konturu]
    # służy do znalezienia największej przerwy między znakami, czyli do przerwy
    # między wyróżnikiem miejsca a wyróżnikiem pojazdu
    char_locations = []
    # kontury znaków, posortowane po składowej x, dzięki czemy możemy zapisać je
    # w odpowiedniej kolejności
    contours, _ = cv2.findContours(chars_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        char = chars_image[y:y + h, x:x + w]
        characters.append(char)
        char_locations.append((x, x+w))
        if debug_folder is not None:
            char_path = os.path.join(debug_folder, f"char_{x}.png")
            cv2.imwrite(char_path, char)

    # znalezienie indeksu największej przerwy między znakami
    max_gap = 0
    max_gap_index = -1
    for i in range(1, len(char_locations)):
        gap = char_locations[i][0] - char_locations[i-1][1]
        if gap > max_gap:
            max_gap = gap
            max_gap_index = i
    return characters, max_gap_index

# utworzenie słownika znaków występujących na polskich rejestracjach
valid_characters = '0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ'
CHARS = {char: index for index, char in enumerate(valid_characters)}

# funkcja zmieniająca wymiary obrazka bez deformacji, domyślne rozmiary odpowiadają
# wymiarą oczekiwanym przez model klasyfikujący znak
def reshape_character(char, target_resized = 216, target_padded = 256):
    height, width = char.shape[:2]
    bigger_dimension = max(height, width)
    resize_ratio = target_resized / bigger_dimension
    if resize_ratio > 1:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    resized = cv2.resize(char, (0,0), fx=resize_ratio, fy=resize_ratio, interpolation=interpolation)
    height, width = resized.shape[:2]
    top = (target_padded-height) // 2
    bottom = target_padded-height-top
    left = (target_padded-width) // 2
    right = target_padded-width-left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = 0)

# extract_characters_from_data(debug=True)
