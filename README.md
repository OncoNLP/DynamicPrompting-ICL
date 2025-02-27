# Dynamic Prompting in Predicting Cancer Outcomes from EHR Notes

This repository contains the framework for reproducing the key learning experiments from our study on **[topic]**. Initially developed for our manuscript submission to **[journal/conference]** in **[month, year]**, this framework is expected to evolve with contributions from the community. Enhancements and updates will be tracked through specific commit checkpoints, and as the codebase improves, results may exhibit slight variations from those in the original study.

## Using This Repository  

This repository provides a structured framework for running machine learning experiments on synthetic and real-world clinical notes. It consists of two main folders:  

#### 1. `synthetic_data` - Simulated Clinical Notes  
This folder contains high-fidelity synthetic data designed to closely replicate the format of UCSF Glioma and Breast Cancer EHR notes. It includes:  
- Full clinical notes with their corresponding labels.  
- Summarized versions of the notes, generated using GPT-4.  
- Cohort coverage: 15 patients each for Glioma and Breast Cancer datasets.  

#### 2. `pipeline` - Experimentation Framework  
This folder contains the core pipeline for executing various learning setups, including:  
- Zero-shot learning, few-shot learning with dynamic prompting, and summarization-enhanced approaches.  
- The *RESULTS.ipynb* notebook, showcasing results from running UCSF Glioma and Breast cancer data on the *Llama 3.0 8B model* and *Llama 3.0 70B model*.  
- Additional documentation in the *README.md* file within the *pipeline* folder, detailing implementation and usage.  

## Contributing to This Work  

We appreciate contributions that help enhance the learning experiments and improve this framework. If you'd like to contribute, please follow these steps:  

1. **Fork the repository** to your GitHub account.  
2. **Create a new branch** for your changes:  
   ```sh
   git checkout -b <your_branch_name>
   ```
3. **Make your modifications** and commit them with a clear message:
   ```sh
   git commit -m "Your commit message"
   ```
4. **Push your changes** to your forked repository:
   ```sh
   git push origin <your_branch_name>
   ```
5. **Submit a pull request (PR)** to the main repository for review.
   

## Contributors

We'd like to thank the following contributors for their valuable work on this project:

- [Shreyas Anil](https://github.com/anils-ucsf)
- [Bhumika Srinivas](https://github.com/srinivasb-ucsf)
- [Hui Lin](https://github.com/hloncai)

And a special thanks to all other contributors who have improved the project over time!

## Contact

For any scientific inquiries about this repository, please contact **[EMAIL]**

## STATEMENT