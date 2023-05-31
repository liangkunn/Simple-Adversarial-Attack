# Simple Adversarial Attack
A simple implementation of adversarial attack on Mnist and CIFAR10, using Fast Gradient Sign Methods (FGSM)


FGSM is showed as below:

<img width="717" alt="Screen Shot 2023-05-31 at 12 28 25 PM" src="https://github.com/liangkunn/SimpleAdversarialAttack/assets/36016499/ed0be318-48b2-46ad-843a-389eb9290eda">

1. Firstly calculates the Loss and then calculate the gradient of the Loss w.r.t the image

2. Then, add the gradient (or its sign for each pixel), multiplied by a small step size, to the original image

3. clamp the modified image to make sure the values of each pixel are between [0,1]

## Visualization of original image and adversarial image for Mnist
Model predicts digit 1 for original image, but predicts 9 if FGSM is implemented. This image is successfully attacked
<img width="846" alt="Screen Shot 2023-05-31 at 12 29 17 PM" src="https://github.com/liangkunn/SimpleAdversarialAttack/assets/36016499/e728692e-af77-4e0c-9bd2-d79b9b4ddad7">

## Visualization of original image and adversarial image for CIFAR10
<img width="864" alt="Screen Shot 2023-05-31 at 12 31 16 PM" src="https://github.com/liangkunn/SimpleAdversarialAttack/assets/36016499/bfe77b75-264e-491f-8a52-27652e86eba2">
