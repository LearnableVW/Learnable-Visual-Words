# Learnable-Visual-Words
Pytorch implementation for paper **Learnable Visual Words for Interpretable Image Recognition**.

# Abstract
To interpret deep models’ predictions, attention-based visual cues are widely used 
in addressing *why* deep models make such predictions. Beyond that, the current
research community becomes more interested in reasoning *how* deep models make
predictions, where some prototype-based methods employ interpretable representa-
tions with their corresponding visual cues to reveal the black-box mechanism of
deep model behaviors. However, these pioneering attempts only either learn the
category-specific prototypes and deteriorate their generalizing capacities, or demon-
strate several illustrative examples without a quantitative evaluation of visual-based
interpretability with further limitations on their practical usages. In this paper,
we revisit the concept of visual words and propose the Learnable Visual Words
(LVW) to interpret the model prediction behaviors with two novel modules: se-
mantic visual words learning and dual fidelity preservation. The semantic visual
words learning relaxes the category-specific constraint, enabling the general visual
words shared across different categories. Beyond employing the visual words for
prediction to align visual words with the base model, our dual fidelity preservation
also includes the attention guided semantic alignment that encourages the learned
visual words to focus on the same conceptual regions for prediction. Experiments
on six visual benchmarks demonstrate the superior effectiveness of our proposed
LVW in both accuracy and model interpretation over the state-of-the-art methods.
Moreover, we elaborate on various in-depth analyses to further explore the learned
visual words and the generalizability of our method for unseen categories.

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
