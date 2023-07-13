# OOD-detection-tool

- This app enables the evaluation of labels predicted by a specific machine learning model on single images.
- The project is a single page web application.
- The whole functionality is implemented in python and the GUI is built with Gradio.
- The answers of the users will be identifiable by unique IDs which correspond to their e-mail addresses.

## Quickstart

#### 0. Just for wsl users:
To get the function of an automation of the path after running the script the following is required (Please be aware that this is just a recommendation. If this step gets skipped the application can even be used without any error): \
Install wslu via PPA: https://launchpad.net/~wslutilities/+archive/ubuntu/wslu \
After, the command "export BROWSER=wslview" needs to be added to the file .bashrc. Then the terminal has to be closed and opened again.

#### 1. Clone repo 
Copy this repo (HTTPS or SSH) and use `git clone 'copied string'` in your terminal to create a local copy of this repo on your machine.

#### 2. Download models
Download the necessary .pt files for the currently available models for this app manually (these are too big to offer them via this github repo). After download they just need to be copied into the root folder of the local copy of this repo. \
Links: https://nc.mlcloud.uni-tuebingen.de/index.php/s/Xgwt7iYb2TrTJy7 (convnext_tiny), https://nc.mlcloud.uni-tuebingen.de/index.php/s/3QizZD7NxgAEpiT (convnext_small)

#### 3. Activate conda environment
This repo provides a .yml file which creates a conda environment. It contains channels and specified versions of dependencies on which this application is built. \
To activate this environment conda needs to be installed first. Then run `conda env create -f ML-OOD-tool.yml` under the project root. Next the environment can be activated by running `conda activate ML-OOD-tool`.

#### 4. Run the application
Run `python3 SPA_Interface.py` under the project root. A link will be generated which has to be copied and pasted in a any browser. Then the first page of the application will be displayed.

## Usage

#### First page:
- Every user will have an own ID with minimum length of four which is used to save the evaluation data of this user.
- Visiting the application for the first time, the user has to enter their e-mail-address. Then their personal ID will be generated.
- The emails_ids.json file stores all these correspondings in a list of JSON object literals in the following way: \
&nbsp; \
&nbsp; <img src="https://github.com/ML-Team-Projekt/OOD-detection-tool/assets/116190225/9ecb80a9-2c34-41ec-b146-79adf0c9a90d)" width="400" height="150">
- The batchsize (amount of images which has to be evaluated) can be choosen. There is also the possibility to decide for the default batchsize of 10.
- The machine learning model can be choosen as well. At the moment the app offers access to convnext_tiny and convnext_small. For the evaluation every image of the randomly generated batch gets passed to the chosen model and its predictions get further processed. 

#### Evaluation page:
- It´s ensured that every user has to evaluate the predictions of one specific image at most once per model.
- At the moment the images are randomly choosen from a set of images within the repo. We also have access to flickerApi but this not public yet. Each image gets rescaled by pytorches interpolation and center-cropped to a size of 256x256.
-  Next to the image there are displayed the top ten out of 1000 labels with their probabilities predicted by the model in decreasing order.
-  For the most likeliest label a short explanation from wikipedia gets displayed as well.
-  Finally on the page can be found three buttons: 'identically and indipendent distributed', 'out of distribution', 'abstain'. With these the user can evaluate if the labels match the image, don´t match the image or unsure.

#### Last page:
- After the evaluation is over, the user can decide if their answers should be saved or not. In case the data should be saved, it gets added to the data.json file. It consists of a list of JSON object literals which all correspond to exactly one image. \
Either a new literal gets added to the list of the key 'UserCall' in one of these or, if a picture has not got evaluated so far, a new JSON object literal gets added to the list. \
The structure in particular: \
&nbsp; \
&nbsp; <img src="https://github.com/ML-Team-Projekt/OOD-detection-tool/assets/116190225/ee999c3c-138d-47f2-91ad-522a6f17e57b" width="600" height="300">

## How to contribute to this repo
- If there shows up a bug or there comes up an idea for extra features, there has to be created an issue and a new branch.
- After working on an issue, there needs be created a PR. At least one approval is acquired to enable merging the changes on main.
- Don´t push directly on the main branch!
