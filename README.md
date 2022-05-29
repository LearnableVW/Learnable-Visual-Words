# Learnable-Visual-Words
Pytorch implementation for paper [Learnable Visual Words for Interpretable Image Recognition].

# Framework
![Alt text](framework.png?raw=true "Title")

# Prerequisites
- PyTorch
- torchvision
- NumPy
- cv2
- PIL
- argparse


# Datasets
Please download the datasets from their official sites: [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html#:~:text=The%20Stanford%20Dogs%20dataset%20contains,of%20fine%2Dgrained%20image%20categorization.), [AWA2](https://cvml.ist.ac.at/AwA2/), [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [STL10](https://cs.stanford.edu/~acoates/stl10/), [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
Prepare the datasets and orgnized all samples as follows:

    ---train
       |--- Class 1
       |--- Class 2
       ...
       |--- Class C

    ---test
       |--- Class 1
       |--- Class 2
       ...
       |--- Class C

# Training The Model:
1. Fine tune the pre-trained ResNet-34 model on the dataset.
2. Extract the attention maps following the instructions of [Grad-Cam](https://github.com/jacobgil/pytorch-grad-cam).
3. Modify the setting file and train the model:
    > python main_lvw.py -train_atts ./data/grad_train_pets.npy -test_atts ./data/grad_test_pets.npy 
4. Evaluate the model by IOU:
    > python evaluation.py -modeldir ./saved_models/resnet34/0034_pets -model 200_19push0.8621.pth -test_atts ./data/grad_test_pets.npy -thr 50

The fine-tuned ResNet-34 model, training and testing Grad-Cam attention for Oxford-IIIT Pet dataset can be download from [Google-Drive](https://drive.google.com/drive/folders/1ibLEWs1H9e4xOOsq5s8C23-P9kYv8bmi?usp=sharing).

# Acknowledgement
This project is built on the open-source implementation [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet).
