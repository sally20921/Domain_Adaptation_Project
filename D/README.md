# SimCLR
```
mkdir output
```


## Training 

```
python train_simclr.py './config/config.yaml'
```

## Linear evaluation

```
python python train_logistic_regression.py './config/config.yaml' output/PATH_TO_GENERATED_TRAINING_OUTPUT 100
```

## Fine tuning

```
python python train_classification.py './config/config.yaml' output/PATH_TO_GENERATED_TRAINING_OUTPUT 100
```


# Results


| Dataset     | Architecture | Batch size | Epochs  | Fine Tuning |
| ------------| ------------ | -----------| ------ |------------ |
| CIFAR10     | ResNet50     | 512        | 100  | 0.78     |

