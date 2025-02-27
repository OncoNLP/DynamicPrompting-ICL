# Dynamic Prompting Pipeline

<p align="center">
  <img src="../images/Dynamic prompting_workflow.drawio.png" width="600" alt="Dynamic Prompting Workflow">
</p>
<p align="center">
  <em>Workflow showcasing the adaptability of the pipeline across zero-shot, few-shot with dynamic prompting, and summarization-enhanced setups for survival prediction using EHR notes.</em>
</p>

# Prerequisites
- **Operating System**:  
  - macOS or Linux (tested on CentOS Stream 10)
  - Windows users may need to use WSL (Windows Subsystem for Linux) for compatibility.

- **Python**:  
  - Python 3.10 or above.

# Folder Structure
```
/pipeline/
│── RESULTS.ipynb         # experiment results on UCSF data
│── breast_metrics.json   # breast cohort results across all folds and experiments
│── glioma_metrics.json   # glioma cohort results across all folds and experiments
│── requirements.txt      # required dependencies
|── run_dynamic_prompt.py # figure pipeline script
```

# Installing
Install the required dependencies using pip and the requirements.txt file:

```sh
pip install -r requirements.txt
```

This will install all the necessary libraries, such as torch, numpy, pandas, and more.

# Running the Script

```
python3 run_dynamic_prompt.py \
    --exp_type zero_shot_summary \
    --fold_number 0 \
    --large False \
    --num_gpus 4 \
    --zero_shot True \
    --examples ../../example_file_0.csv \
    --test_data ../../test_file_0.csv \
    --summary True \
    --cancer_type breast
```

# Argument Descriptions

| Argument       | Type  | Description |
|---------------|------|-------------|
| `--exp_type`  | `str`  | Type of experiment (`zero_shot_summary`, `zero_shot_fn`, `dynamic_summary`, `dynamic_fn`) |
| `--fold_number` | `int`  | Cross-validation fold number (default: `0`) |
| `--large` | `bool` | Use the large model if `True`, else use the small model (default: `False`) |
| `--num_gpus` | `int`  | Number of GPUs for parallelism (default: `1`) |
| `--zero_shot` | `bool` | Use zero-shot learning if `True`, else dynamic prompting (default: `False`) |
| `--examples` | `str`  | Path to CSV file with training examples |
| `--test_data` | `str`  | Path to CSV file with test samples |
| `--summary` | `bool` | Use summarized text if `True`, else full clinical notes (default: `True`) |
| `--vector_db` | `str`  | Path to vector database pickle file (optional) |
| `--cancer_type` | `str`  | Cancer type (`glioma` or `breast cancer`) |


# Data
The `RESULTS.ipynb` notebook utilizes the UCSF Glioma and Breast dataset to demonstrate the results of running the Python script. The dataset consists of unstructured EHR (Electronic Health Record) notes from 475 Glioma patients and 503 Breast cancer patients, recorded 180 days post-diagnosis. The data is split into training and testing sets, with 80% used for training and 20% reserved for testing.

For those interested in using the synthetic data generated as part of this project, please refer to the README file located in the `synthetic_data` folder for further instructions on how to access and utilize it.

# STATEMENT
