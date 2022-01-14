# DCASE2020 task 1b -- Low-Complexity Acoustic Scene Classification

## First to do
> \$ cd train
> \$ ./download-dataset.sh
> \$ conda env create -f environment.yml

## How to use

### Model training
To train Mobnet/small-FCNN, please run
> \$ cd train  
> \$ ./train.sh 

### Quantization
To compress well-trained model by quantization, please run
> \$ cd quantization  
> \$ python model_trans.py  

### Evaluation
To evaluate trained models and quantized models, please run
> \$ cd eval  
> \$ ./eval.sh  
 

## Pre-trained models
Pre-trained models are provided in `./pretrained_models`, including
* Mobnet (before and after quantization)
* small-FCNN (before and after quantization)
 
 
## Environment 

- Note the post-training dynamic range quantization currently supports tensorflow 2.3. Please update the version of tensorflow as below.


```shell
 pip uninstall -y tensorflow
 pip install -q tf-nightly
 pip install -q tensorflow-model-optimization
```

Please also check the [official document](https://www.tensorflow.org/model_optimization/guide/quantization/training_example) for a latest version updates.
