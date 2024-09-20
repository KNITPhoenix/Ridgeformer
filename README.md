# Ridgeformer: a Cross Domain Fingerprint Matching Network

### Code for paper under review in ICASSP 2025

![fingerprint_main_dia](https://github.com/user-attachments/assets/5bdb4d46-5cea-4ca2-a001-8406689543cb)

## Installations and environment creation
- conda create -n ridgeformer python=3.8.19
- conda activate ridgeformer
- pip install -r requirements.txt

It requires timm version 0.5.0 and can be installed with the given .whl file.
It also requires pytorch>=2.2.2

## Preparing data and pretrained checkpoints

### Datasets used in training and their application link
- [Ridgebase(RB)](https://www.buffalo.edu/cubs/research/datasets/ridgebase-benchmark-dataset.html#title_0)
- [The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database Version 1.0](http://www4.comp.polyu.edu.hk/~csajaykr/fingerprint.htm)
- [IIITD SmartPhone Fingerphoto Database V1 (ISPFDv1)](https://iab-rubric.org/index.php/ispfdv1)
- [IIITD SmartPhone Finger-Selfie Database V2 (ISPFDv2)](https://iab-rubric.org/index.php/ispfdv1)

### Testing dataset:
- The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database Version 1.0 (HKPolyU)
- Ridgebase (RB)

### Preprocessing ISPFD v1 and v2 datasets
- scipts in ISPFD_preprocessing directory are used to segment out contactless images in ISPFD dataset
- requires SAM checkpoint and openai clip
- can be used after installing [segment-anything](https://pypi.org/project/segment-anything/) and downloading SAM [checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- For more information, refer to SAM's official repository [Link](https://github.com/facebookresearch/segment-anything)

### Manifest files creation for dataloaders
- script data_folder_creation.py in datasets directory is used to arrange the datasets in a specific folder structure

  Subject -> finger -> background and instances
  
- script manifest_file_creation.py in datasets directory is used to create manifest files used in dataloaders. The manifest file structure will be as follows:

'''
{
Unique_finger_id_1:{
'Contactless': ( list of paths of all contactless images )
'Contactbased': ( list of paths of all contactbased images )
},
Unique_finger_id_2:{
'Contactless': ( list of paths of all contactless images )
'Contactbased': ( list of paths of all contactbased images )
}
......}
'''

### Pretrained models and Finetuned checkpoints
Download the zip file from [Link](https://buffalo.box.com/s/8wmvwhmvbmfsy8j7lr7ppa30bxe3hvws) and unzip the contents in ridgeformer_checkpoints directory to use in evaluation and training scripts

## Training
Stage 1 - train_combined.py is used to train the model on Stage 1 of our architecture

Stage 2 - train_combined_fusion.py is used to train the model on Stage 2 of our architecture

All the performance ROCs and matrices are saved in combined_models_scores directory

All tensorboard logs are saved in experiment_logs directory

## Testing and Evaluation
### HKPolyU
- Evaluation of HKPolyU testing dataset on finetuned checkpoint from Stage 1 can be done using hkpoly_evaluation_phase1.py
- Evaluation of HKPolyU testing dataset on finetuned checkpoint from Stage 2 can be done using hkpoly_evaluation_phase2.py

### Ridgebase
- Evaluation of ridgebase testing dataset on pretrained checkpoint from Stage 1 can be done using rb_evaluation_phase1.py
- Evaluation of ridgebase testing dataset on pretrained checkpoint from Stage 2 can be done using rb_evaluation_phase1.py

## Performance compared with SOTA methods on HKPolyU dataset (1:1 verification)
|Method | Probe | Gallery | EER(%) | TAR(%)@FAR=.01 |
| :---: | :---: | :---: | :---: | :---: |
|Verifinger | CL | CB | 19.31 | 76.00 |
|RTPS+DCM | CL | CB | 14.33 | 50.50 |
|Multi-Siamese | CL | CB | 7.93 | 54.00 |
|MANet | CL | CB | 4.13 | 88.50 |
|ML Fusion | CL | CB | 4.07 | 94.40|
|Ridgeformer (Ours)| CL | CB | 2.83 | 89.34|

## Performance compared with SOTA methods on HKPolyU dataset (1:N identification)
|Method | Probe | Gallery | R@1 | R@10 |
| :---: | :---: | :---: | :---: | :---: |
|Verifinger | CL | CB | 80.73 | 91.00 |
|RTPS+DCM | CL | CB | 66.67 | 83.00 |
|Multi-Siamese | CL | CB | 64.59 | 91.00 |
|MANet | CL | CB | 83.54 | 97.00 |
|Ridgeformer (Ours)| CL | CB | 87.40 | 98.23 |

## Performance compared with SOTA methods on Ridgebase dataset (1:1 verification)
|Method | Probe | Gallery | EER(%) | TAR(%)@FAR=.01 |
| :---: | :---: | :---: | :---: | :---: |
|Verifinger | CL | CB | 18.90 | 57.60 |
|Ridgeformer (Ours) | CL | CB | 5.25 | 82.23 |
|AdaCos(CNN) | CL | CL | 21.30 | 61.20 |
|Verifinger | CL | CL | 19.70 | 63.30 |
|Ridgeformer (Ours)| CL | CL | 7.60 | 85.14 |

## Performance compared with SOTA methods on Ridgebase dataset (1:N identification)
|Method | Probe | Gallery | R@1 | R@10 |
| :---: | :---: | :---: | :---: | :---: |
|Verifinger | CL | CB | 72.50 | 89.20 |
|Ridgeformer (Ours) | CL | CB | 69.90 | 92.64 |
|Verifinger | CL | CL | 85.20 | 91.40 |
|AdaCos(CNN) | CL | CL | 81.90 | 89.50 |
|Ridgeformer (Ours)| CL | CL | 100.00 | 100.00 |

## Contact
For more information or any questions, feel free to reach us at spandey8@buffalo.edu

## License
Ridgeformer is CC-BY-NC 4.0 licensed, as found in the LICENSE file. It is released for academic research / non-commercial use only.
