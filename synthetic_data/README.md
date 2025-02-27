# Synthetic Data

## Overview
This folder contains synthetic data generated for testing, development, and benchmarking purposes. The data is designed to mimic UCSF Glioma and Breast cancer patient notes while ensuring privacy and compliance.

## Folder Structure
```
/synthetic_data/
│── GPT_templates_breast.csv # Summary notes for breast
│── GPT_templates_glioma.csv # Summary notes for glioma
│── patient_notes_breast.csv # Full notes for breast
│── patient_notes_glioma.csv # Full notes for glioma
```

## Data Description
This dataset contains the following fields:

| Column Name      | Description                                      |
|------------------|--------------------------------------------------|
| `note`           | clinical notes / clinical summary                |
| `label`          | binary outcome label                             |

- **Format**: CSV
- **Size**: 15 Glioma patients and 5 Breast cancer patients

## Data Generation Process
The data was generated using UCSF Versa (in-house GPT-4o) to mimic the format and structure of UCSF data.

## ⚠️ Disclaimer
This dataset is **purely synthetic** and does not contain any real user information. It is intended for testing and research purposes only and should not be used in production environments.
