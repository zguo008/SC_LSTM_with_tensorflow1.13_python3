# SC_LSTM_with_tensorflow1.13_python3

An implementation of SC_LSTM_with_tensorflow1.13_python3 This repo is an improvement of [this implementation](https://github.com/hit-computer/SC-LSTM). This repo also shows a way to prepare a parallel dataset using gensim. 
## Preparing dataset
NTU_DATA/create_word_embedding/preprocess_csv.py and create_embedding.py<br />

## Getting Started

To train a two-tier model : comment out samplernn/model_new_A.py line 293, 294 ./run_4_16.sh <br />
To train a one-tier model : comment out samplernn/model_new_A.py line 293, 295 ./run_16_16.sh<br />
To train a baseline model : comment out samplernn/model_new_A.py line 294, 295 ./run_16.sh
 
### Prerequisites

Tensorflow 1.13 and above; python 3.7

## Running the tests
To generate sound using two-tier model: change the logdir in generate_4_16.sh then ./generate_4_16.sh<br />

To generate sound using one-tier model: change the logdir in generate_16_16.sh then ./generate_16_16.sh<br />
To generate sound using baseline model: change the logdir in generate_16.sh then ./generate_16.sh<br />
