# Pneumonia Detection using Chest XRays leveraging CNNs

This project aims to detect and classify pneumonia using chest X-rays through Convolutional Neural Networks (CNNs). The goal is to improve pneumonia diagnosis, especially in resource-constrained healthcare settings, where accurate and timely diagnosis is crucial.

## Introduction

Pneumonia is a prevalent respiratory infection, often requiring chest X-rays for diagnosis. However, skilled interpretation and resource shortages can lead to delays in diagnosis and treatment. This project explores using CNNs for automated pneumonia detection.

## Methodology

### Datasets

Three datasets were used, covering different lung conditions:

1. [Chest X-ray Images (Pneumonia) by Paul Mooney (2018)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

2. [COVID-19 X-ray (2 proposed databases) by Edoardo Vantaggiato (2020)](https://www.kaggle.com/edoardovantaggiato/covid19-xray-two-proposed-databases)

3. [NIH Chest X-rays by National Institutes of Health (NIH) (2017)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
     
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

We will train the model for 30 epochs with a batch size of 30 using stochastic gradient descent (SGD) optimizer with a learning rate of 0.001 and a momentum of 0.9. We will use the cross-entropy loss function and monitor the validation accuracy during training.

1. Loading datasets using the steps in the .ipynb file load dataset from kaggle.

```
train_data = datasets.ImageFolder(root=f"/kaggle/input/covid19-xray-two-proposed-databases/Datasets/5-classes/Train", transform=transform)
val_data = datasets.ImageFolder(root=f"/kaggle/input/covid19-xray-two-proposed-databases/Datasets/5-classes/Val", transform=transform)
test_data = datasets.ImageFolder(root=f"/kaggle/input/covid19-xray-two-proposed-databases/Datasets/5-classes/Test", transform=transform)
```

2. Using transform function to pre-process and apply transformations on the dataset.

```
transform = v2.Compose([
    v2.Resize(256),   #must same as here
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
```

3. Split the dataset into 60:25:15

```
train_size = int(0.6 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset)-train_size-val_size
```

4. Create train/validation loaders.

```
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
```

5. Pass it to the epoch loop for training and validation.

#### Running Pre-trained Model

#### Source Code

#### Dataset
 
#### Acknowledgements

#### Contributors

| **Team Members**        | **Student ID** | **Email**                  |
|-------------------------|----------------|----------------------------|
| Vandit Mehta            | 40232414       | mehtavandit2205@gmail.com |
| Konark Shah             | 40232277       | Konarkshah2010@gmail.com  |
| Urvilkumar Patel        | 40230630       | patelurvil38@gmail.com     |
| Anurag Teckchandani     | 40263724       | anuragteckchandani@gmail.com |

| **Lecturer**            | **Email**                  |
|-------------------------|----------------------------|
| Dr. Mahdi S. Hosseini   | mahdi.hosseini@concordia.ca |

| **Teaching Assistant**  | **Email**                  |
|-------------------------|----------------------------|
| Ahmed Alagha (Lead TA)  | ahmed.alagha@mail.concordia.ca |

