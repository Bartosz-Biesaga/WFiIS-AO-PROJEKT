# Detekcja aut i odczytywanie rejestracji

___

### Autorzy

- [Bartosz Biesaga](https://github.com/Bartosz-Biesaga)
- [Paweł Dyjak](https://github.com/paweld37)
- [Julia Zwierko](https://github.com/juliazwierko)

### Uruchamianie aplikacji

#### Wymagania
GUI zostało przygotowane z wykorzystaniem pakietu tkinter, który w większości przypadków jest dostarczany razem z pythonem, w przypadku gdyby nie był on dostępny można go zainstalować korzystając z polecenia:
```shell
sudo apt-get install python3-tk
```
Aplikacja była tworzono i testowana z python 3.12.3.

#### Instalacja pakietów i uruchomienie aplikacji w wirtualnym środowisku:
```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_with_deps1.txt
pip3 install -r requirements_with_deps2.txt
pip3 install -r requirements_without_deps.txt --no-deps
python3 main/gui.py
```
W folderze *test_data* znajdują się zdjęcia, na których można przetestować działanie aplikacji.

### Struktura projektu

#### Folder *main*
W tym folderze umieściliśmy wszystkie pliki potrzebne do działania projeku, czyli skrypty odpowiedzialne za interfejs użytkownika (*gui.py*), detekcję (*YOLO_utils.py*), segmentację (*segmentation.py*) oraz rozpoznawanie znaków (*recognition.py*).

#### Folder *models*
W tym folderze umieściliśmy notebooki, które uruchamialiśmy w google colab, w celu nauki modelu YOLO i do rozpoznawania pojedynczych znaków oraz wagi tych modeli z najlepszymi wynikami osiągniętymi w czasie treningu na zbiorach walidacyjnych, aby móc je załadować w pozostałych skryptach.

#### Folder *dokumentacja*
W tym folderze umieściliśmy dokumentację naszego projektu.

#### Foldery *test_data* oraz *results*
Są to foldery na odpowiednio dane do testowania aplikacji oraz na zapisywanie rezultatów.
