# Knowledge Distillation for Urban Sound Tagging

## Problem Statement
Apply knowledge distillation to the [Urban Sound Tagging task (Task 5)](http://dcase.community/challenge2019/task-urban-sound-tagging) of the DCASE 2019 contest.

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
Two approaches were reviewed and implemented for knowledge distillation:
1. <b>[Softmax activation based](https://arxiv.org/abs/1503.02531)  knowledge distillation</b>
2. <b>[Similarity preserving](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf) knowledge distillation </b> 

## Results

| Teacher model | Student model | BCE Loss | BCE + KD loss | BCE + KD + SP Loss |
|---|---|---|---|---|
| DCASE Net | DCASE Small | 0.751 | 0.762 | 0.772 |
| DCASE Net | CNN-LSTM | 0.680 | 0.692 | 0.673 |
| MobileNetV2 (1.0) | MobileNetV2 (0.0005) |0.659 | 0.687||

## References

