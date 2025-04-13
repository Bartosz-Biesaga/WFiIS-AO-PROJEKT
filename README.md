# Car detection and license plate recognition
Project for the fifth semester Image Analysis course at AGH.
___
### Project description
The project is a simple desktop application, that allows the user to automatically detect vehicles and license 
plates in the loaded image and then read detected license plates.

Project was written in Python using ultralytics package for vehicle and license plate detection, opencv for 
license plate character segmentation, pytorch for character recognition and tkinter for GUI.

### Authors
- [Bartosz Biesaga](https://github.com/Bartosz-Biesaga)
- [Paweł Dyjak](https://github.com/paweld37)
- [Julia Zwierko](https://github.com/juliazwierko)
  
### Starting the app

#### Requirements
GUI was created using tkinter package, which in most cases is delivered with python, but in case it is not avaiable it may be installed with following command:
```shell
sudo apt-get install python3-tk
```
Application was developed and tested using python 3.12.3.

#### Installing dependencies and starting the app in virtual environment
```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_with_deps1.txt
pip3 install -r requirements_with_deps2.txt
pip3 install -r requirements_without_deps.txt --no-deps
python3 main/gui.py
```
In *test_data* folder there are images that may be used to test the application.

### Folder structure

#### *main*
All source files are located there. GUI scripts in *gui.py*, detection in *YOLO_utils.py*, segmentation in *segmentation.py* and character recognition in *recognition.py*.

#### *models*
All notebooks that we run in google colab to train YOLO and character recognition model and their weights saved from training when their performance on validation datasets were best to easily load them in other scripts.

#### *dokumentacja*
Project documentation (written in Polish).

#### *test_data* and *results*
Folders for respectively data for app testing and for saving the results.
