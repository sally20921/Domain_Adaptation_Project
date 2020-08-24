# SimCLR


## Training 

```
python train_simclr.py PATH_TO_CONFIG_FILE
```

## Linear evaluation

```
python python train_logistic_regression.py PATH_TO_CONFIG_FILE output/PATH_TO_GENERATED_TRAINING_OUTPUT EPOCH_NUM
```

## Fine tuning

```
python python train_classification.py PATH_TO_CONFIG_FILE output/PATH_TO_GENERATED_TRAINING_OUTPUT EPOCH_NUM
```


# Results

_Evaluated accuracy for the test set. Evaluation performed after training the model using simclr._

| Dataset     | Architecture | Batch size | Epochs | Linear Evaluation | Fine Tuning |
| ------------| ------------ | -----------| ------ | ----------------- |------------ |
| CIFAR10     | ResNet50     | 512        | 1000   | 0.7957            | 0.7828      |

