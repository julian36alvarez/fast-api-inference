# fast_api_inference 

By: ECI.

Version: 0.1.0

A short description of the project.

## Prerequisites

- [Anaconda](https://www.anaconda.com/download/) >=4.x
- Optional [Mamba](https://mamba.readthedocs.io/en/latest/)

## Create environment

```bash
conda env create -f environment.yml
activate fast_api_inference
```

or 

```bash
mamba env create -f environment.yml
activate fast_api_inference
```

## Project organization

    fast_api_inference
        ├── data
        │   ├── processed      <- The final, canonical data sets for modeling.
        │   └── raw            <- The original, immutable data dump.
        │
        ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
        │                         the creator's initials, and a short `-` delimited description, e.g.
        │                         `julian-alvarez-v`.
        │
        ├── .gitignore         <- Files to ignore by `git`.
        │
        ├── environment.yml    <- The requirements file for reproducing the analysis environment.
        │
        └── README.md          <- The top-level README for developers using this project.

---
Project created for demonstration


docker run -p 8000:8000 -v //c/Users/Julian-Alvarez/.aws:/root/.aws -e ENDPOINT_NAME=nombre-de-tu-endpoint -e BUCKET_LOAD=awss3-sagemakerawscdkparamosurveillancecolombiafbf-lnpp3i43hza5 -e BUCKET_OUTPUT=awss3-inputoutputparamosurveillancecolombiaaee921c-fmronsdwgz8o fast-api