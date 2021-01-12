# Knowledge Distillation for Urban Sound Tagging

## Problem Statement
Apply knowledge distillation to the Urban Sound Tagging task (Task 5) of the DCASE 2019 contest.

## How to Run
1. Run `make_mel.py` to generate the mel spectrograms
2. Run `train_teacher.py` to train the teacher model and generate its weights
3. Run `train_student.py` to train the student model

## Models and Hyperparameters
The models and hyperparameters for the teacher and student can be changed from the `config.ini` file. <br>
Options for the teacher model include: <br>
1. `dcase_net`
2. `mobilenetv2` 
Select the required value of width multiplier for `mobilenetv2`
Options for the student model include: <br>
1. `dcase_small`
2. `cnn_lstm`
3. `mobilenetv2`

## Implemented Approaches

## Results

## References

