# Dynamic Medical Imaging and Computing Lab

Welcome to the Dynamic Medical Imaging and Computing Lab! Research can be an amazing experience and propel your career forward. However, you cannot approach it like a traditional class. To be successful, you have to learn to break down your overarching question into subquestions and find the answer yourself.

## How to Find Your Own Answers

1. Google
2. Ask ChatGPT → guide to AI Prompting
3. Ask other undergrads (slack)
4. Ask the Ph.D. students
5. Ask the post-doc
6. Ask Dr. Castillo

## Lab Directory

- Jorge Cisneros (post-doc)
- Patrick Giolando (post-grad)
  - Sid (undergrad)
- Yi-Kuan Liu (post-grad)
  - Caleb (undergrad)
  - Aaron (undergrad)
  - Lizzy (undergrad)
- Amanda (grad)
  - Gabriela (undergrad)
- Ananya (undergrad)
- Taima (undergrad)
- Laura (undergrad)

## Repo Directory
- yi-kuan-pft  - COPD Progression Classification with XGBoost
- Aaron-Segmentation - Lung and lobe segmentation with UNET-based models

## Resources

### Potentially useful repos
- [MONAI - Models, Loss Functions, and More](https://github.com/Project-MONAI)
- [Facebook Segment Anything Model](https://github.com/facebookresearch/segment-anything)


### General

- [Intro to machine learning (concepts only ~20min)](https://www.youtube.com/watch?v=IpGxLWOIZy4)
- [Machine-learning dictionary](https://developers.google.com/machine-learning/glossary)
- [Free intro course - translated from Chinese, from dans 214L](https://www.coursera.org/learn/machine-learning)
- [Paid intro to ML course - from Stanford prof. $49/month](https://www.coursera.org/specializations/machine-learning)
- [Machine Learning crash course to get you up and running](https://developers.google.com/machine-learning/crash-course)
- [Good website to go in depth on certain topics](https://machinelearningmastery.com/start-here)
- Python Machine Learning package: TensorFlow
  - To import in Python: `import tensorflow as tf`
  - [Resource on how to write a program using TensorFlow](https://keras.io/api/)
  - [Learn PyTorch in a Day (25-hour long video)](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=11463s) 
- XGBoost:
  - [Easy XGBoost tutorial](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)
  - [Parameter Tuning](https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/)
  - [Tuning hyperparameters of an Estimator - sci kit learn](https://scikit-learn.org/stable/modules/grid_search.html)
  - [Evaluate performance of classification model - ROC Curve and AUC](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

### Beginner Feedforward Stuff

- [Overall Vid](https://youtu.be/QK7GJZ94qPw)
- [PyTorch Video](https://youtu.be/oPhxf2fXHkQ)
- Text/Documentation:
  - [PyTorch Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
  - [Deep Learning Wizard - PyTorch Feedforward Neural Network](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/)

### MATLAB

- Video:
  - [MATLAB Neural Networks](https://youtu.be/6T2yYTSw8z0)
  - [MATLAB Feedforward Neural Network](https://youtu.be/-98SQYpCjvg)
- Text/Documentation:
  - [Create and Train a Feedforward Neural Network](https://www.mathworks.com/help/thingspeak/create-and-train-a-feedforward-neural-network.html)

### Segmentation
- [Original UNET Paper](https://arxiv.org/abs/1505.04597)
- [UNETR](https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/document/9706678)
- [SWin UNET](https://arxiv.org/pdf/2111.14791v2.pdf)

### Re-installing NVIDIA drivers and CUDA on Linux
- Purge Drivers
  - sudo apt-get remove --purge '^nvidia-.*'
  - sudo apt-get autoremove

- [CUDA Archive, check pytorch stable versions](https://developer.nvidia.com/cuda-toolkit-archive)
- Select OS Version and architecture (Usually Ubuntu x86_64)
- deb(network)
- run commands up to sudo apt-get -y install cuda
- run sudo apt-get -y install cuda-XX-X
  - e.g. sudo apt-get -y install cuda-11-8

If you get a safeboot screen, run systemctl reboot --firmware-setup
Boot settings, turn off safe boot
Verify installation with nvidia-smi
