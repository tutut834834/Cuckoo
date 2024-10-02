# Cuckoo - Clean Label Backdoor on Horizontal Federated Learning Systems

Code for our paper, "Cuckoo Attacks: Clean Label Backdoor on Horizontal Federated Learning Systems". This repository provides the implementation and experimental evaluation of clean label backdoor attacks in horizontal federated learning (HFL) environments. The attack leverages malicious triggers embedded in training data that appear legitimate, allowing adversaries to covertly compromise global models.

## Abstract

In this research, we investigate clean label backdoor attacks in the context of horizontal federated learning (HFL). Federated learning enables distributed training across multiple clients while preserving privacy by keeping local data on devices. However, this framework is vulnerable to adversarial backdoor attacks, where malicious data appears clean to evade detection. We implement and adapt clean label backdoor strategies to the HFL paradigm, demonstrating how local malicious clients can covertly introduce triggers into the global model without altering their data labels.

Our experimental evaluation shows that, even under privacy-preserving constraints, clean label backdoor attacks effectively compromise global models, leading to misclassification during inference. Existing defense mechanisms are largely ineffective in detecting these attacks in a federated learning setup. The results underline the challenge of securing HFL models against covert adversaries.

A public repository for this project can be found under [link to repository].

![image](https://github.com/user-attachments/assets/b6e09556-7e42-4a54-b31f-c034dea08706)


## Installation

All required packages are listed in `requirements.txt`. This can be installed in a virtual environment using tools such as `virtualenv` or `conda`.

Example of installation via `pip`:

```bash
pip install -r requirements.txt
