# Crossmodal knowledge distillation hashing（CKDH）

## Install

Our work follows the relevant configuration requirements of [ViLBERT](https://github.com/facebookresearch/vilbert-multi-task), 
please read the README.md of ViLBERT for more information.

## DataSets

* Download the datasets. for example:[MIRFLICKR](https://press.liacs.nl/mirflickr/mirdownload.html) and [COCO](https://cocodataset.org/#home)
* For the method of extracting picture features, please refer to [ViLBERT](https://github.com/facebookresearch/vilbert-multi-task)'s Data Setup.

## Train
* Finetune the network with the cross-entropy loss for classification to accelerate training convergence. Such as:<br>
```
python pre_teacher_main.py --from_pretrained <pretrained_model_path>
```
* Teacher NetWork: set the "from_pretrained" and start training.
```
python train_teacher.py --from_pretrained <pretrained_model_path>
```
* Student NetWork: set the teacher network model path and start training.
```
python train_student.py --from_pretrained <teacher_network_model_path>
```

## Citation
If you find this work is helpful in your research, please cite:<br>
```
@article{wang2022ckdh,
  title={Crossmodal knowledge distillation hashing},
  author={Wang, Jinhui and Jin, Lu and Li, Zechao and Tang, Jinhui},
  journal={SCIENTIA SINICA Technologica},
  year={2022},
  doi={10.1360/SST-2021-0214}
}
```

## Acknowledgement
Thanks to [ViLBERT](https://github.com/facebookresearch/vilbert-multi-task) for their help in this work.
