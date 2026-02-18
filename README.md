# Finger tapping Predictor

## Description

This repository is the official implementation of: (future link to the paper)

This repository provides an experimental web app for Parkinson's disease assessment using finger tapping videos.

## Dataset

For testing this web-based application, you can use our dataset for FIS videos is available at https://zenodo.org/records/17738775. 

This dataset contains 234 video recordings of controls and patients (individuals diagnosed with Parkinson’s disease) performing the standardized finger tapping test, a commonly used motor assessment in clinical and research settings. The videos were collected as part of a collaborative study conducted by researchers from the University of Burgos and the Hospital Universitario de Burgos. This work was supported by the project PI19/00670 of the Ministerio de Ciencia, Innovación y Universidades, Instituto de Salud Carlos III, Spain.

The dataset is intended to support research on motor symptom characterization, quantitative assessment methods, and the development of automated analysis tools for Parkinson’s disease. All recordings were obtained following appropriate ethical approvals, and participants provided informed consent for research use of their data.

All the participants and the Parkinson's Disease Association are thanked for their support. Similarly, we thank Dr. Gamez Leiva and Dr. Madrigal for the videos assessments.

The structure of the zip file is:

- fis_diagnostic.csv. Table with two columns, the first one is the name of the video and the second one the UPDRS rating for the clip.
- videos. Folder with 234 videos (left and right hands).

## Code Information

This code has been written in Python and a Jupyter Notebook is also used for running the pipeline. In te sections below, you can find the requirements and how to use this code.

## Requirements

You can create a conda environment with the needed packages running:

```setup
conda create --name new_env --file requirements.txt
```

## Usage Instructions 

For running the application in your environment, I would suggest two options, depending the OS or your environment features. In any case, you should invoke "run.py" file.

a) Option 1: Invoking python directly

```setup
python.exe run.py
```

b) Option 2: Using gunicorn

```setup
nohup gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 600 "run:create_app()" > output.log 2>&1 &
```

## References

If you use this code in your research, please cite our paper

```
Pending
```