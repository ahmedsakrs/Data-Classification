# Data-Classification
Building multi models to classify numerical data of Gamma, Hadron dataset.
## Steps of Applying Data Classification
### Data Balancing
It was done by undersampling to let the two classes be balanced.
### Data Splitting
The dataset was split into
1. 70% train
2. 30% test
### Data Scaling
It is noticed that the ranges of the features are not close, so MinMaxScaler is applied on the data.
### Modeling
We trained all the following machine learning algorithms:
1. Decision Tree Classifier
</br>No hyperparameter tuning was done.</br>The metrics for Decision Tree are as the following:
    * Accuracy= 79.09
    * Precision= 79.19
    * Recall= 78.91
    * F1 score= 79.05
    * Specificity= 79.26
2. Naive Bayes
</br>No hyperparameter tuning was done.</br>The metrics for Naive-Bayes are as the following:
    * Accuracy= 63.71
    * Precision= 59.20
    * Recall= 88.19
    * F1 score= 70.84
    * Specificity= 39.23

    The poor performance for Naive-Bayes is because of the naive assumption that the features are independent, but the data description shows that the features are dependent and nearly extracted from each others.
3. K-Nearest Neighbors (KNN)
</br>K-Fold Cross Validation was used to tune the value of K.</br>The metrics for KNN are as the following:
    * Accuracy= 81.58
    * Precision= 77.90
    * Recall= 88.19
    * F1 score= 82.73
    * Specificity= 74.98
4. AdaBoost Classifier
</br>K-Fold Cross Validation was used to tune the value of n-estimators.</br>The metrics for AdaBoost are as the following:
    * Accuracy= 81.93
    * Precision= 81.79
    * Recall= 82.15
    * F1 score= 81.97
    * Specificity= 81.70
5. Random Forest Classifier
</br>K-Fold Cross Validation was used to tune the value of n-estimators.</br>The metrics for Random Forest are as the following:
    * Accuracy= 86.44
    * Precision= 84.88
    * Recall= 88.68
    * F1 score= 86.74
    * Specificity= 84.20
6. PyTorch Double Layer ANN
</br> The architecture of the neural network is shown in the following block:
```
class BinaryClassification(nn.Module):
    def __init__(self, neurons1, neurons2):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(10, neurons1) 
        self.layer_2 = nn.Linear(neurons1, neurons2)
        self.layer_out = nn.Linear(neurons2, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(neurons1)
        self.batchnorm2 = nn.BatchNorm1d(neurons2)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
```
   K-Fold Cross Validation is used to tune the number of neurons in the first hidden layer and the second hidden layer.</br>The metrics for Random Forest are as the following:</br>
* Accuracy= 85.74
* Precision= 89.27
* Recall= 81.26
* F1 score= 85.08
* Specificity= 90.23
