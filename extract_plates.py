import os
import xml.etree.ElementTree as ET
from PIL import Image


if __name__ == "__main__":

    xml_file = "annotations.xml"  # Plik XML z danymi o tablicach
    images_folder = "images"  # Folder z obrazami
    target_folder = "plates"  # Folder na wycięte tablice rejestracyjne
    output_txt_file = "plates_data.txt"  # Plik tekstowy na dane tablic

    os.makedirs(target_folder, exist_ok=True)

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
                print(f"Obraz {image_name} nie istnieje, pomijam.")
                continue

            with Image.open(image_path) as img:
                for box in image.findall(".//box[@label='plate']"):
                    # Pobierz współrzędne tablicy rejestracyjnej
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))

                    # Wytnij tablicę rejestracyjną z obrazu
                    cropped = img.crop((xtl, ytl, xbr, ybr))

                    # Zapisz wyciętą tablicę do pliku z oryginalną nazwą
                    cropped_path = os.path.join(target_folder, image_name)
                    cropped.save(cropped_path)

                    # Pobierz numer rejestracyjny
                    plate_number = box.find(".//attribute[@name='plate number']").text

                    # Zapisz dane do pliku tekstowego
                    output_file.write(f"{image_name} {plate_number}\n")
                    print(f"Zapisano wyciętą tablicę: {cropped_path}")

    print("Proces wycinania tablic zakończony.")
