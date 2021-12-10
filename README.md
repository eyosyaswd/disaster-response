disaster-response
==============================

Implementation and improvement of the paper 'Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response'

```bib
@inproceedings{multimodalbaseline2020,
  Author = {Ferda Ofli and Firoj Alam and Muhammad Imran},
  Booktitle = {17th International Conference on Information Systems for Crisis Response and Management},
  Keywords = {Multimodal deep learning, Multimedia content, Natural disasters, Crisis Computing, Social media},
  Month = {May},
  Organization = {ISCRAM},
  Publisher = {ISCRAM},
  Title = {Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response},
  Year = {2020}
}
```

----------------------------------

## Instructions

To run the code:
1. Download dataset related files:
    - Go to https://crisisnlp.qcri.org/crisismmd and download "CrisisMMD dataset version v2.0" and "Datasplit for multimodal baseline results with agreed labels." 
    - Go to https://crisisnlp.qcri.org/lrec2016/lrec2016.html and download "Word2vec embeddings trained using crisis-related tweets".
    - Save the unzipped version of the downloaded folders into disaster-response/data/raw/. They should have the same folder names as shown below. 
2. Download the required libraries (listed in conda_requirements.yml). You can use the following command if you want a conda environment: 
```
conda create -n disaster-response -f conda_requirements.yml
```
3. Build features (modality = [text, image]): 
```
cd src/features/
python build_[modality]_features.py
```
4. Train models (modality = [text, image, multimodal]):
```
cd src/models/
python train_[modality]_model.py
```
5. Test models (modality = [text, image, multimodal]):
```
cd src/models/
python predict_[modality]_model.py
```
Preprocessed data will be outputted in data/interim/, predictions will be outputted in data/processed/, and performance metrics will be outputted in reports/.
You can specify the task ("humanitarian" or "informative") in each of the .py files mentioned above. The text model can be run on CPU, but a GPU is highly recommended to run the image or multimodal models. 

----------------------------------

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- Instructions on how to run the code.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- Final predictions of the test data.
    │   └── raw            <- The original, immutable data dump.
    |       ├── crisismmd_datasplit_agreed_label
    |       ├── CrisisMMD_v2.0
    |       └── crisisNLP_word2vec_model
    │
    ├── models             <- Trained models. 
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Implemented paper can be found here.
    │
    ├── reports            <- Generated analysis. 
    |                         Written report and performance metrics of models found here.
    │
    ├── conda_requirements.yml   <- The requirements file for reproducing the analysis environment. 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_[modality]_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions. Also scripts for creating data generators or NN architectures.
            ├── custom_dataset.py 
            ├── predict_[modality]_model.py
            ├── sentence_cnn.py 
            └── train_[modality]_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
