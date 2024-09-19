# Ridgeformer: a Cross Domain Fingerprint Matching Network

### Code for paper under review in ICASSP 2025

![fingerprint_main_dia](https://github.com/user-attachments/assets/5bdb4d46-5cea-4ca2-a001-8406689543cb)

## Installations and environment creation
- conda create -n ridgeformer python=3.8.19
- conda activate ridgeformer
- pip install -r requirements.txt

# Prepare data and pretrained checkpoints

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
- script manifest_file_creation.py in datasets directory is used to create manifest files used in dataloaders  

### Pretrained models and Finetuned checkpoints: [Link](https://buffalo.box.com/s/8wmvwhmvbmfsy8j7lr7ppa30bxe3hvws)
Download the zip file and unzip the contents in ridgeformer_checkpoints directory to use in evaluation and training scripts

## License
Ridgeformer is CC-BY-NC 4.0 licensed, as found in the LICENSE file. It is released for academic research / non-commercial use only.
