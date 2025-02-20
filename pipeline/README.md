# Dynamic Prompting Pipeline

<div style="text-align: center;">
  <img src="../images/Dynamic prompting_workflow.drawio.png" width="600" alt="Dynamic Prompting Workflow" style="display: block; margin: auto;">
  <p><em>Workflow showcasing the adaptability of the pipeline across zero-shot, few-shot with dynamic prompting, and summarization-enhanced setups for survival prediction using EHR notes.</em></p>
</div>

# Prerequisites
- **Operating System**:  
  - macOS or Linux (tested on CentOS Stream 10)
  - Windows users may need to use WSL (Windows Subsystem for Linux) for compatibility.

- **Python**:  
  - Python 3.10 or above.

# Installing
Install the required dependencies using pip and the requirements.txt file:

```sh
pip install -r requirements.txt
```

This will install all the necessary libraries, such as torch, numpy, pandas, and more.

# Instructions

About the script and running it.

# Data
The `RESULTS.ipynb` notebook utilizes the UCSF Glioma dataset to demonstrate the results of running the Python script. The dataset consists of unstructured EHR (Electronic Health Record) notes from 475 Glioma patients, recorded 180 days post-diagnosis. The data is split into training and testing sets, with 80% used for training and 20% reserved for testing.

For those interested in using the synthetic data generated as part of this project, please refer to the README file located in the `synthetic_data` folder for further instructions on how to access and utilize it.


# STATEMENT
