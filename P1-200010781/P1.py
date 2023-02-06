#!/usr/bin/env python
# coding: utf-8
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns


def readData():
    df = pd.read_csv("drug_consumption.data.csv", header=None)
    # Delete the first index column
    df.drop(df.columns[0], axis=1, inplace=True)
    # Delete the middle useless columns
    df.drop(df.columns[12:28], axis=1, inplace=True)
    # Delete the last two columns
    df.drop(df.columns[-1], axis=1, inplace=True)
    df.drop(df.columns[-1], axis=1, inplace=True)

    return df


def dataChecking(df):
    # Check data types
    print("Types: \n", df.dtypes)
    # Check the number of null values
    print("Number of NaN: \n", df.isnull().sum())
    # Check if there is "?" values
    print("Check if '?' in samples: \n", '?' in df)
    # Check if there are duplicated columns
    print("Check if duplicated columns exist: \n", set(df.duplicated()))


def plot(df, outputData):
    # plot between each pairs
    sns.pairplot(df, hue=29, height=2.5)

    # Show the distribution of all classes
    x = sorted(list(set(outputData)))
    plt.bar(x, outputData.value_counts()[x])
    plt.show()


def splitData(inputData, outputData):
    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(inputData, outputData, test_size=0.2, random_state=0,
                                                        stratify=outputData)
    # split training data into real train and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0,
                                                      stratify=y_train)

    print("x_train shape: {}".format(x_train.shape))
    print("x_test shape: {}".format(x_test.shape))
    print("x_val shape: {}".format(x_val.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("y_test shape: {}".format(y_test.shape))
    print("y_val shape: {}".format(y_val.shape))
    return x_train, x_test, x_val, y_train, y_test, y_val


def scaling(x_train, x_val, x_test):
    # Feature scaling
    # Fit on training set
    x_scaler = StandardScaler().fit(x_train)
    # Transform on all three set
    x_scaler_train = x_scaler.transform(x_train)
    x_scaler_val = x_scaler.transform(x_val)
    x_scaler_test = x_scaler.transform(x_test)
    return x_scaler_train, x_scaler_val, x_scaler_test
    # print("x_train means: \n", x_scaler_train.mean(axis=0))
    # print("x_train std: \n", x_scaler_train.std(axis=0))
    # print("x_val means: \n", x_scaler_val.mean(axis=0))
    # print("x_val std: \n", x_scaler_val.std(axis=0))
    # print("x_test means: \n", x_scaler_test.mean(axis=0))
    # print("x_test std: \n", x_scaler_test.std(axis=0))


def training1(x_scaler_train, x_scaler_val, y_train, y_val):
    # Fitting logistic regression to the training set using unbalanced class weight
    Classifier = LogisticRegression(penalty='none', class_weight=None)
    Classifier.fit(x_scaler_train, y_train)

    # predict the training set
    y_pred_train = Classifier.predict(x_scaler_train)
    print("\nModel for training set with None class_weight")
    print("Accuracy score: ", accuracy_score(y_train, y_pred_train))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_train, y_pred_train))
    print("confusion_matrix: \n", confusion_matrix(y_train, y_pred_train))

    print("precision_score(average=None): \n", precision_score(y_train, y_pred_train, average=None, zero_division=0))
    print("precision_score(average=micro): ", precision_score(y_train, y_pred_train, average='micro'))
    print("precision_score(average=macro): ", precision_score(y_train, y_pred_train, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_train, y_pred_train, average=None))
    print("recall_score(average=micro): ", recall_score(y_train, y_pred_train, average='micro'))
    print("recall_score(average=macro): ", recall_score(y_train, y_pred_train, average='macro'))

    # predict the validation set
    y_pred_val = Classifier.predict(x_scaler_val)
    print("\nModel for validation set with None class_weight")
    print("Accuracy score: ", accuracy_score(y_val, y_pred_val))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_val, y_pred_val))
    print("confusion_matrix: \n",
          confusion_matrix(y_val, y_pred_val, labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]))

    print("precision_score(average=None): \n", precision_score(y_val, y_pred_val, average=None, zero_division=0))
    print("precision_score(average=micro): ", precision_score(y_val, y_pred_val, average='micro'))
    print("precision_score(average=macro): ", precision_score(y_val, y_pred_val, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_val, y_pred_val, average=None))
    print("recall_score(average=micro): ", recall_score(y_val, y_pred_val, average='micro'))
    print("recall_score(average=macro): ", recall_score(y_val, y_pred_val, average='macro'))

    return Classifier


def training2(x_scaler_train, x_scaler_val, y_train, y_val):
    # Fitting logistic regression to the training set using balanced class weight
    Classifier1 = LogisticRegression(penalty='none', class_weight='balanced')
    Classifier1.fit(x_scaler_train, y_train)

    # predict the training set
    y_pred_train1 = Classifier1.predict(x_scaler_train)
    print("\nModel for training set with balanced class_weight")
    print("Accuracy score: ", accuracy_score(y_train, y_pred_train1))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_train, y_pred_train1))
    print("confusion_matrix: \n",
          confusion_matrix(y_train, y_pred_train1, labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]))

    print("precision_score(average=None): \n", precision_score(y_train, y_pred_train1, average=None, zero_division=0))
    print("precision_score(average=micro): ", precision_score(y_train, y_pred_train1, average='micro'))
    print("precision_score(average=macro): ", precision_score(y_train, y_pred_train1, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_train, y_pred_train1, average=None))
    print("recall_score(average=micro): ", recall_score(y_train, y_pred_train1, average='micro'))
    print("recall_score(average=macro): ", recall_score(y_train, y_pred_train1, average='macro'))

    # predict the validation set
    y_pred_val1 = Classifier1.predict(x_scaler_val)
    print("\nModel for validation set with balanced class_weight")
    print("Accuracy score: ", accuracy_score(y_val, y_pred_val1))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_val, y_pred_val1))
    print("confusion_matrix: \n",
          confusion_matrix(y_val, y_pred_val1, labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]))

    print("precision_score(average=None): \n", precision_score(y_val, y_pred_val1, average=None, zero_division=0))
    print("precision_score(average=micro): ", precision_score(y_val, y_pred_val1, average='micro'))
    print("precision_score(average=macro): ", precision_score(y_val, y_pred_val1, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_val, y_pred_val1, average=None))
    print("recall_score(average=micro): ", recall_score(y_val, y_pred_val1, average='micro'))
    print("recall_score(average=macro): ", recall_score(y_val, y_pred_val1, average='macro'))

    return Classifier1


def evaluation1(Classifier, x_scaler_test, y_test):
    # predict the testing set using unbalanced class weight
    y_pred_test = Classifier.predict(x_scaler_test)
    print("\nModel for testing set with None class_weight")
    print("Accuracy score: ", accuracy_score(y_test, y_pred_test))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_test, y_pred_test))
    print("confusion_matrix on testing set: \n",
          confusion_matrix(y_test, y_pred_test, labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]))

    print("precision_score(average=None): \n", precision_score(y_test, y_pred_test, average=None, zero_division=0))
    print("precision_score(average=micro): \n", precision_score(y_test, y_pred_test, average='micro'))
    print("precision_score(average=macro): \n", precision_score(y_test, y_pred_test, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_test, y_pred_test, average=None))
    print("recall_score(average=micro): \n", recall_score(y_test, y_pred_test, average='micro'))
    print("recall_score(average=macro): \n", recall_score(y_test, y_pred_test, average='macro'))


def evaluation2(Classifier1, x_scaler_test, y_test):
    # predict the testing set using balanced class weight
    y_pred_test1 = Classifier1.predict(x_scaler_test)
    print("\nModel for testing set with balanced class_weight")
    print("Accuracy score: ", accuracy_score(y_test, y_pred_test1))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_test, y_pred_test1))
    print("confusion_matrix on testing set: \n",
          confusion_matrix(y_test, y_pred_test1, labels=["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]))

    print("precision_score(average=None): \n", precision_score(y_test, y_pred_test1, average=None, zero_division=0))
    print("precision_score(average=micro): \n", precision_score(y_test, y_pred_test1, average='micro'))
    print("precision_score(average=macro): \n", precision_score(y_test, y_pred_test1, average='macro'))
    print("recall_score(average=None): \n", recall_score(y_test, y_pred_test1, average=None))
    print("recall_score(average=micro): \n", recall_score(y_test, y_pred_test1, average='micro'))
    print("recall_score(average=macro): \n", recall_score(y_test, y_pred_test1, average='macro'))


def polyFeatures(data):
    print("\n Polynomial Features:\n")
    poly = PolynomialFeatures(degree=2)
    newData = poly.fit_transform(data)
    # print(newData)
    print("Number of New Features: " + str(poly.n_output_features_))
    return newData


def regTraining1(newData_train, newData_val, y_train, y_val):
    # Fitting logistic regression to the training set using none class weight and L1 reg
    regClassifier1 = LogisticRegression(penalty='l1', class_weight=None, solver='liblinear')
    regClassifier1.fit(newData_train, y_train)

    # predict the training set
    newy_pred_train = regClassifier1.predict(newData_train)
    print("\nModel for training set with None class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_train, newy_pred_train))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_train, newy_pred_train))

    # predict the validation set
    newy_pred_val = regClassifier1.predict(newData_val)
    print("\nModel for validation set with None class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_val, newy_pred_val))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_val, newy_pred_val))

    return regClassifier1


def regTraining2(newData_train, newData_val, y_train, y_val):
    # Fitting logistic regression to the training set using balanced class weight and L1 reg
    regClassifier2 = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
    regClassifier2.fit(newData_train, y_train)

    # predict the training set
    newy_pred_train = regClassifier2.predict(newData_train)
    print("\nModel for training set with balanced class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_train, newy_pred_train))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_train, newy_pred_train))

    # predict the validation set
    newy_pred_val = regClassifier2.predict(newData_val)
    print("\nModel for validation set with balanced class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_val, newy_pred_val))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_val, newy_pred_val))

    return regClassifier2


def regEvaluation1(regClassifier1, newData_test, y_test):
    # predict the testing set using unbalanced class weight and L1 reg
    newy_pred_test = regClassifier1.predict(newData_test)
    print("\nModel for testing set with None class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_test, newy_pred_test))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_test, newy_pred_test))


def regEvaluation2(regClassifier2, newData_test, y_test):
    # predict the testing set using balanced class weight and L1 reg
    newy_pred_test = regClassifier2.predict(newData_test)
    print("\nModel for testing set with balanced class_weight and L1 regularisation")
    print("Accuracy score: ", accuracy_score(y_test, newy_pred_test))
    print("balanced_accuracy_score: ", balanced_accuracy_score(y_test, newy_pred_test))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # read dataset
    df = readData()
    # check data types and check if exists null value or duplicated data
    dataChecking(df)

    # get input features and output targets
    inputData = df.iloc[:, 0:12]  # features
    outputData = df[29]  # targets

    # plot data and classification distribution
    plot(df, outputData)

    # split data into training, validation and testing sets
    x_train, x_test, x_val, y_train, y_test, y_val = splitData(inputData, outputData)
    # data scaling
    x_scaler_train, x_scaler_val, x_scaler_test = scaling(x_train, x_val, x_test)

    # First training with class_weight=none
    Classifier = training1(x_scaler_train, x_scaler_val, y_train, y_val)
    # Second training with class_weight=balanced
    Classifier1 = training2(x_scaler_train, x_scaler_val, y_train, y_val)

    # Evaluate testing set using model with class_weight=none
    evaluation1(Classifier, x_scaler_test, y_test)
    # Evaluate testing set using model with class_weight=balanced
    evaluation2(Classifier1, x_scaler_test, y_test)

    # ------Using new data to train------
    # Produce polynomial features
    newData_train = polyFeatures(x_scaler_train)
    newData_val = polyFeatures(x_scaler_val)
    newData_test = polyFeatures(x_scaler_test)

    regClassifier1 = regTraining1(newData_train, newData_val, y_train, y_val)
    regClassifier2 = regTraining2(newData_train, newData_val, y_train, y_val)

    regEvaluation1(regClassifier1, newData_test, y_test)
    regEvaluation2(regClassifier2, newData_test, y_test)
