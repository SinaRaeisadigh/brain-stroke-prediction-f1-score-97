# brain-stroke-prediction-f1-score-97

### 1. **Import Libraries**
The code imports necessary libraries for data manipulation, model building, evaluation, and handling class imbalance. This includes `pandas` for data handling, `sklearn` for model training and evaluation, and `imblearn` for SMOTE to deal with imbalanced datasets.

### 2. **Data Preprocessing**
The dataset is loaded, and certain columns are transformed. The 'ever_married' column is encoded as binary (1 for 'Yes', 0 for others), and the 'gender' column is also converted to binary (1 for Female, 0 for others). Categorical columns such as 'work_type', 'Residence_type', and 'smoking_status' are converted into dummy variables (one-hot encoding), creating separate binary columns for each category.

### 3. **Splitting Data into Training and Test Sets**
The data is divided into features (`X`) and the target variable (`y`). The target variable represents whether the person had a stroke or not. The SMOTE algorithm is applied to the dataset to oversample the minority class (stroke cases) to balance the data.

After that, the data is split into training and testing sets using `train_test_split`, where 75% of the data is used for training and 25% for testing. The shapes of the resampled and split data are printed to verify the split.

### 4. **Random Forest Classifier**
A Random Forest classifier is initialized with 100 trees and balanced class weights to account for class imbalance. The model is trained on the resampled training data.

### 5. **Predictions and Model Evaluation**
The model makes predictions on the test data. The evaluation metrics are:
- **Accuracy**: The percentage of correct predictions.
- **Classification Report**: It provides precision, recall, F1-score for each class (stroke vs no stroke), which helps assess how well the model is performing across different metrics.
- **Confusion Matrix**: It shows the counts of true positives, true negatives, false positives, and false negatives, which are useful for analyzing misclassification patterns. 
