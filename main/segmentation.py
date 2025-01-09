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

def extract_characters_from_data(debug = False):
    xml_file = "annotations.xml"  # Plik XML z danymi o tablicach
    images_folder = "photos"  # Folder z obrazami
    target_folder = "characters"  # Folder na wycięte znaki z tablic rejestracyjnych
    output_txt_file = "labels.txt"  # Plik tekstowy na etykiety znaków

    os.makedirs(target_folder, exist_ok=True)
    # statystyki na temat tego ile tablic rejestracyjnych udało się odczytać z dobrą
    # ilościa znaków
    total_plates = 0
    well_read_plates = 0
    # statystyki na temat tego ile jakich znaków na ile możliwych udało się
    # uzyskać ze zbioru danych
    chars_count = [0 for _ in CHARS]
    total_chars_count = [0 for _ in CHARS]

    # Otwórz plik tekstowy do zapisu danych o tablicach
    with open(output_txt_file, "w", encoding="utf-8") as output_file:
        # Wczytaj i sparsuj plik XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iteracja po elementach <image> w pliku XML
        for image in root.findall(".//image"):
            image_name = image.get("name")
            image_path = os.path.join(images_folder, image_name)

            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            for box in image.findall(".//box[@label='plate']"):
                # Pobierz współrzędne tablicy rejestracyjnej
                xtl = int(float(box.get("xtl")))
                ytl = int(float(box.get("ytl")))
                xbr = int(float(box.get("xbr")))
                ybr = int(float(box.get("ybr")))

                # Wytnij tablicę rejestracyjną z obrazu
                cropped = img[ytl:ybr, xtl:xbr]

                # Pobierz numer rejestracyjny
                plate_number = box.find(".//attribute[@name='plate number']").text.replace(" ", "")
                # poprawienie błędnej etykiety
                if image_name == "58.jpg":
                    plate_number = "STA8582C"
                total_plates += 1
                for char in plate_number:
                    total_chars_count[CHARS[char]] += 1
                # nazwa pliku bez rozszerzenia
                simple_image_name = image_name[:-4]
                try:
                    if debug == True:
                        processed_license_plate = process_image(img = cropped,
                                    debug_folder = f"debug/{simple_image_name}")
                    else:
                        processed_license_plate = process_image(img = cropped)
                except ValueError as e:
                    print(f"{e}, {image_name}")
                    continue
                if debug == True:
                    characters, space_location = get_characters_images(processed_license_plate,
                                            debug_folder = f"debug/{simple_image_name}/chars")
                else:
                    characters, space_location = get_characters_images(processed_license_plate)

                if len(characters) != len(plate_number):
                    if debug == True:
                        print(f"Extracted {len(characters)} characters in {image_name}, should be {len(plate_number)}, skipping")
                    continue
                well_read_plates += 1

                # Zapisz wycięte znaki
                for i, character in enumerate(characters):
                    character_path = os.path.join(target_folder, f"{simple_image_name}_{i}.png")
                    cv2.imwrite(character_path, reshape_character(character))
                    #cv2.imwrite(character_path, character)
                    output_file.write(f"{simple_image_name}_{i}.png, {CHARS[plate_number[i]]}\n")
                    chars_count[CHARS[plate_number[i]]] += 1
    if debug == True:
        print(f"Udało się odczytać prawidłową liczbę znaków z {well_read_plates/total_plates*100:.0f}% rejestracji")
        for char, count, total_count in zip(CHARS, chars_count, total_chars_count):
            print(f"{char}: {count}/{total_count}")

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