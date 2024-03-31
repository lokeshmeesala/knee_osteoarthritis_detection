# Detecting Early Stage Knee Osteoarthritis Using Deep Transfer Learning

<b>Abstract</b> — Knee osteoarthritis is one of the most prevalent forms of the disease, and its diagnosis can be challenging, especially in its early stages. Imaging techniques such as X-Ray are commonly used to diagnose osteoarthritis, but the interpretation of these images can be subjective and prone to error, especially when detecting  subtle changes. In this research, I aim to develop a deep learning network that can classify Knee X-ray images into 5 categories, i.e. 0 - Normal/Healthy, 1 - Doubtful, 2 - Minimal, 3 - Moderate, and 4 - Severe. I propose to use Convolutional Neural Networks (CNN) for multi-class image classification. The baseline model will be a CNN-based deep learning network, which will be trained on a dataset of knee X-ray images. The effectiveness of transfer learning is investigated by applying state-of-the-art CNN architectures such as ResNet, and VGG Nets. To handle the class imbalance, a selective augmentation technique is used. An iterative model training process is used for fine-tuning. 

<b><i>Keywords</i></b> — Osteoarthritis, CNN, VggNet, ResNets, Multi-class Image Classification, Transfer Learning, Selective Augmentation.

<b>Implementation Details</b>
  - Transfer learning from ResNet and fine tune it by adding custom layers.
  - Selective Augmentation is implemented to deal with the class imbalance problem.
  - To improve the fine tuning results, iterative model training is implemented.
  - Achieved an F1 score of 62.3%.

<b>Research Paper</b><br>
Results and conclusions of this project are mentioned in this research paper.
[https://github.com/lokeshmeesala/knee_osteoarthritis_detection/blob/main/docs/Final%20Project%20Report.pdf]

<b>Dataset</b><br>
Dataset of knee X-ray images from the Osteoarthritis Initiative (OAI) database is used. Labelling is done by Chen, Pingjun (2018) [https://data.mendeley.com/datasets/56rmx5bjcr].
