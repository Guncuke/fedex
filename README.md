# fedex
 a simple federate-learning framework write by python
![fedex](https://github.com/Guncuke/fedex/assets/78009909/cc66a1d8-837c-4e3e-93d0-bb4c14c24caf)

How to run?
> streamlit run main.py

## 1. Client

Client-side native training code, just like normal deep learning model training

## 2. Server

The server is responsible for aggregating the received client model parameters, and the user can customize the desired aggregation strategy

## 3. Dataset

Define your own dataset in the dataset.py

## 4. Model

Define your own model in models.py

## 5. distribution

Currently the average and Dirichlet distributions are implemented
