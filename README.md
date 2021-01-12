# Knowledge Distillation for Urban Sound Tagging

## Problem Statement
Apply knowledge distillation to the [Urban Sound Tagging task (Task 5)](http://dcase.community/challenge2019/task-urban-sound-tagging) of the DCASE 2019 contest. Since the focus is on model compression, only the coarse grained classification task has been considered. 

## How to Run
1. Run `make_mel.py` to generate the mel spectrograms
2. Run `train_teacher.py` to train the teacher model and generate its weights
3. Run `train_student.py` to train the student model

## Config
The models and hyperparameters for the teacher and student can be changed from the `config.ini` file.
Options for the teacher model include: <br>
1. `dcase_net`
2. `mobilenetv2` 

Options for the student model include: <br>
1. `dcase_small`
2. `cnn_lstm`
3. `mobilenetv2`

Select the required value of width multiplier wherever `mobilenetv2` is used

## Implemented Approaches
Two approaches were reviewed and implemented for knowledge distillation:
1. <b>[Softmax activation based](https://arxiv.org/abs/1503.02531)  knowledge distillation</b>
2. <b>[Similarity preserving](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf) knowledge distillation </b> 

## Models used
1. DCASE Net
This model is taken from the [third placed team](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kim_107.pdf) of the DCASE 2019 contest.
2. DCASE Small
A model similar to DCASE Net but with lesser number of channels per layer.
3. CNN-LSTM
A network consisting of one CNN layer followed by a single layer LSTM cell. This model has been described in [this](https://indico2.conference4me.psnc.pl/event/35/contributions/2907/) paper.
4. MobileNetV2
The torchvision implementation of MobileNetV2 modified to accept single channel images as input.

## Results

| Teacher model | Student model | BCE | BCE + KD | BCE + KD + SP |
|---|---|---|---|---|
| DCASE Net | DCASE Small | 0.751 | 0.762 | <b>0.772</b> |
| DCASE Net | CNN-LSTM | 0.680 | <b>0.692</b> | 0.673 |
| MobileNetV2 (1.0) | MobileNetV2 (0.0005) |0.659 | 0.667 |<b>0.687</b>|

From the table, it can be observed that similarity preserving knowledge distillation seems to be effective when both the student and the teacher model have similar model architectures.

<p align="center">
	<img src='https://github.com/hardhat5/knowledge-distillation-ust/blob/master/images/activation_map.jpg'/>
</p>

## References
1. [DCASE 2019 Task 5: Urban Sound Tagging](http://dcase.community/challenge2019/task-urban-sound-tagging)
2. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
3. [Similarity Preserving Knowledge Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf)
4. [Intra-Utterance Similarity Preserving Knowledge Distillation](https://indico2.conference4me.psnc.pl/event/35/contributions/2907/)
5. [Convolutional Neural Networks with Transfer Learning for Urban Sound Tagging](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kim_107.pdf)
6. [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)

