from ultralytics import YOLO
import cv2 as cv
import torch
import os

def crop_boxes_from_image(yolo, image, license_plate_car_ioa=0.85, confidence=0.25,
                          iou = 0.7, save_prediction = False):
    result = yolo(image, conf=confidence, iou=iou, verbose=False)[0]
    if save_prediction is True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(dir_path, "..", "results",
                                 "prediction.jpg")
        result.save(filename=save_path)
    cars = []
    license_plates = []
    for cls, box in zip(map(int, map(torch.Tensor.tolist, result.boxes.cls)),
                        [list(map(int, box)) for box in map(torch.Tensor.tolist, result.boxes.xyxy)]):
        if cls == 0:
            # [wyciete auto, bounding box]
            cars.append([image[box[1]:box[3], box[0]:box[2]], box])
        else:
            # [wycieta tablica rejestracyjna, bounding box, pole powierzchni rejestracji,
            # czy rejestracja zostala przypisana do jakiegos auta]
            license_plates.append([image[box[1]:box[3], box[0]:box[2]], box,
                                   (box[2] - box[0]) * (box[3] - box[1]), False])

    # usuwanie wykrytych dwukrotnie tablic rejestracyjnych, poprzez wybranie wiekszej
    indices_to_remove = []
    for i in range(len(license_plates)):
        license_plate_i_left_top_x = license_plates[i][1][0]
        license_plate_i_left_top_y = license_plates[i][1][1]
        license_plate_i_bottom_right_x = license_plates[i][1][2]
        license_plate_i_bottom_right_y = license_plates[i][1][3]
        for j in range(i+1, len(license_plates)):
            x = (min(license_plate_i_bottom_right_x, license_plates[j][1][2])
                 - max(license_plate_i_left_top_x, license_plates[j][1][0]))
            y = (min(license_plate_i_bottom_right_y, license_plates[j][1][3])
                 - max(license_plate_i_left_top_y, license_plates[j][1][1]))
            # jesli tablic sie pokrywaja, usuwamy mniejsza z nich
            if x > 0 and y > 0:
                if license_plates[i][2] > license_plates[j][2]:
                    indices_to_remove.append(j)
                else:
                    indices_to_remove.append(i)
    for index in indices_to_remove:
        license_plates.pop(index)


    # lista indeksow aut do ktorych przypisano dokladnie jedna rejestracje
    single_license_plate_car_indices = []
    # lista indeksow aut do ktorych przypisano ponad jedna rejestracje
    multiple_license_plate_car_indices = []
    # lista list indeksow tablic rejestracyjnych przypisanych do danego auta
    car_license_plate_indices_list = []
    for i, car in enumerate(cars):
        #lista indeksow tablic rejestracyjnych przypisanych do danego auta
        license_plate_indices = []
        for j, license_plate in enumerate(license_plates):
            x = min(license_plate[1][2], car[1][2]) - max(license_plate[1][0], car[1][0])
            y = min(license_plate[1][3], car[1][3]) - max(license_plate[1][1], car[1][1])
            # sprawdzenie czy stosunek pola powierzchni iloczynu bounding boxa
            # rejestracji oraz bounding boxa auta do pola powierzchni rejestracji
            # przekracza zadany prog w celu przypisania danej rejestracji do
            # danego auta
            if x > 0 and y > 0 and ((x * y / license_plate[2]) >= license_plate_car_ioa):
                license_plate_indices.append(j)
                license_plate[3] = True
        car_license_plate_indices_list.append(license_plate_indices)
        if len(license_plate_indices) == 1:
            single_license_plate_car_indices.append(i)
        if len(license_plate_indices) > 1:
            multiple_license_plate_car_indices.append(i)

    # proba usuniecia nieprawidlowego przypisania rejestracji w sytuacji gdzie
    # w bounding box'ie danego auta znajduje sie wiele rejestacji.
    # Algorytm usuwa z aut o wielu przypisanych rejestracjach te, ktore
    # rownoczesnie przypisane sa do innego auta jako jedyne
    # Algorytm nie gwarantuje usuniecia wszystkich nieprawidlowo przypisanych
    # rejestracji od auta
    while len(multiple_license_plate_car_indices) >= 1 and len(single_license_plate_car_indices) >= 1:
        single_license_plate_car_index = single_license_plate_car_indices.pop()
        single_license_plate_index = car_license_plate_indices_list[single_license_plate_car_index][0]
        for multiple_license_plate_car_index in multiple_license_plate_car_indices:
            try:
                car_license_plate_indices_list[multiple_license_plate_car_index].remove(single_license_plate_index)
            except ValueError:
                continue
            if len(car_license_plate_indices_list[multiple_license_plate_car_index]) == 1:
                single_license_plate_car_indices.append(multiple_license_plate_car_index)
                multiple_license_plate_car_indices.remove(multiple_license_plate_car_index)

    # [wyciete auto, lista wycietych tablic rejestracyjnych przypisanych do auta]
    pairs = []
    for i, car in enumerate(cars):
        license_plate_list = [license_plates[j][0] for j in car_license_plate_indices_list[i]]
        pairs.append([car[0], license_plate_list])

    not_attached_license_plates = []
    for license_plate in license_plates:
        if license_plate[3] is False:
            not_attached_license_plates.append(license_plate[0])
    pairs.append([None, not_attached_license_plates])
    return pairs

# przyklad uzycia
if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    yolo = YOLO(os.path.join(dir_path, "..", "models", "YOLO", "weights", "best.pt"))
    image = cv.imread(os.path.join(dir_path, "..", "test_data", "test_image_14.jpg"))
    pairs = crop_boxes_from_image(yolo, image, save_prediction=True)
    for i, pair in enumerate(pairs):
        car, license_plate_list = pair
        if car is not None:
            cv.imwrite(os.path.join(dir_path, "..", "results", f"car_{i}.jpg"), car)
            for j, license_plate in enumerate(license_plate_list):
                cv.imwrite(os.path.join(dir_path, "..", "results",
                                        f"license_plate_{i}_{j}.jpg"), license_plate)