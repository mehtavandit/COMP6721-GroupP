# Pneumonia Detection using Chest XRays leveraging CNNs

This project aims to detect and classify pneumonia using chest X-rays through Convolutional Neural Networks (CNNs). The goal is to improve pneumonia diagnosis, especially in resource-constrained healthcare settings, where accurate and timely diagnosis is crucial.

## Introduction

Pneumonia is a prevalent respiratory infection, often requiring chest X-rays for diagnosis. However, skilled interpretation and resource shortages can lead to delays in diagnosis and treatment. This project explores using CNNs for automated pneumonia detection.

## Methodology

### Datasets

Three datasets were used, covering different lung conditions:
1. Dataset 1: Pneumonia vs. Normal (5,863 images)
2. Dataset 2: Covid-19, bacterial pneumonia, viral pneumonia, lung opacity, and normal cases (3,555 images)
3. Dataset 3: NIH Chest X-rays dataset, 14 classes representing various lung diseases (large collection)

### CNN Models

Three CNN architectures were employed:
1. ResNet18
2. DenseNet121
3. InceptionV3

### Optimization Algorithm

Stochastic Gradient Descent (SGD) optimizer was used with a learning rate of 0.001, batch size of 30, and cross-entropy loss function.

### Evaluation metrics used for assessing model performance:
- Accuracy
- Precision
- Recall
- F1-score
  
## Results

- InceptionV3 consistently outperformed ResNet-18 and DenseNet-121 across all datasets.
- Higher accuracy and metric scores were observed with fewer classes, suggesting simpler tasks facilitate better feature learning.

## Conclusion

InceptionV3 showed robust learning and generalization capabilities across datasets. However, its computational complexity poses challenges for resource-constrained environments.

#### Requirements


#### Training and Validation

#### Running Pre-trained Model

#### Source Code

#### Dataset
 
#### Acknowledgements

#### Contributors

### Team Members
1. Vandit Mehta
   - Student ID: 40232414
   - Email: mehtavandit2205@gmail.com
2. Konark Shah
   - Student ID: 40232277
   - Email: Konarkshah2010@gmail.com
3. Urvilkumar Patel
   - Student ID: 40230630
   - Email: patelurvil38@gmail.com
4. Anurag Teckchandani
   - Student ID: 40263724
   - Email: anuragteckchandani@gmail.com

### Lecturer
- Dr. Mahdi S. Hosseini
  - Email: mahdi.hosseini@concordia.ca

### Teaching Assistant
- Ahmed Alagha (Lead TA)
  - Email: ahmed.alagha@mail.concordia.ca

