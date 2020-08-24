
python packages:
[fire](https://github.com/google/python-fire) for commandline api

## Install

```bash
git clone --recurse-submodules (this repo)
cd $REPO_NAME/code
(use python >= 3.5)
pip install -r requirements.txt
```
Place the data folder at `$REPO_NAME/data`.


## Data Folder Structure
```
code/
  cli.py
  train.py
  evaluate.py
  infer.py
  ...
data/
```


## How to Use

### Training

```bash
cd code
python cli.py train
```

- Access the prompted tensorboard port to view basic statistics.
- At the end of every epoch, a checkpoint file will be saved on `data/ckpt/OPTION_NAMES`

For further configurations, take a look at `code/config.py` and
[fire](https://github.com/google/python-fire).

### Evaluation

```bash
cd code
python cli.py evaluate --ckpt_name=$CKPT_NAME
```
e\.g\. `--ckpt_name=model_name_dmm_ckpt_3/loss_0.4818_epoch_15`

### Making submissions
```bash
python cli.py infer --ckpt_name=$CKPT_NAME
```

The above command will save the outcome at the prompted location. 
<!-- To get answers from validation data split, change `--split test` to `--split val`. -->

<!--
### Evaluating submissions

```bash
cd code/scripts
python eval_submission.py -y $SUBMISSION_PATH -g $DATA_PATH
```
-->
