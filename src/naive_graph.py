from cProfile import label
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from my_naive_bayes import MyNaiveBayes as MNB 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys


def main():
    df = pd.read_csv('../data/heart_disease_dataset_UCI.csv')
    standardScalar = StandardScaler()
    target_columns = ['age','trestbps','chol','thalach','oldpeak']
    df[target_columns] = standardScalar.fit_transform(df[target_columns])
    X= df.drop(['target'], axis=1)
    Y= df['target']
    testSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    sklearn_accuracy = [None] * len(testSizes)
    sklearn_precision = [None] * len(testSizes)
    sklearn_recall = [None] * len(testSizes)
    sklearn_tn = [None] * len(testSizes)
    sklearn_fp = [None] * len(testSizes)
    sklearn_fn = [None] * len(testSizes)
    sklearn_tp = [None] * len(testSizes)
    
    my_accuracy = [None] * len(testSizes)
    my_precision = [None] * len(testSizes)
    my_recall = [None] * len(testSizes)
    my_tn = [None] * len(testSizes)
    my_fp = [None] * len(testSizes)
    my_fn = [None] * len(testSizes)
    my_tp = [None] * len(testSizes)

    for testIndex in range(0, len(testSizes)):
        #sklearn implementation
        X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=testSizes[testIndex], random_state=40)
        gaussBayes = GaussianNB()
        gaussModel = gaussBayes.fit(X_train, Y_train)
        predictionGaussBayes = gaussModel.predict(X_test)
        cmGaussBayes = confusion_matrix(Y_test, predictionGaussBayes)
        tn, fp, fn, tp = cmGaussBayes.ravel()
        sklearn_accuracy[testIndex] = accuracy_score(Y_test, predictionGaussBayes)
        sklearn_precision[testIndex] = precision_score(Y_test, predictionGaussBayes)
        sklearn_recall[testIndex] = recall_score(Y_test, predictionGaussBayes)
        sklearn_tn[testIndex] = tn
        sklearn_fp[testIndex] = fp
        sklearn_fn[testIndex] = fn
        sklearn_tp[testIndex] = tp
        #my implementation
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()
        Y_train_np = Y_train.to_numpy()
        Y_test_np = Y_test.to_numpy()
        gaussBayes = MNB()
        gaussBayes.fit(X_train_np, Y_train_np)
        predictionGaussBayes = gaussBayes.predict(X_test_np)
        cmGaussBayesm = confusion_matrix(Y_test_np, predictionGaussBayes)
        # print(*cmGaussBayesm)
        tnm, fpm, fnm, tpm = cmGaussBayesm.ravel()
        my_accuracy[testIndex] = accuracy_score(Y_test_np, predictionGaussBayes)
        my_precision[testIndex] = precision_score(Y_test_np, predictionGaussBayes)
        my_recall[testIndex] = recall_score(Y_test_np, predictionGaussBayes)
        my_tn[testIndex] = tnm
        my_fp[testIndex] = fpm
        my_fn[testIndex] = fnm
        my_tp[testIndex] = tpm
        # print(tpm)
    # print('testSizes')
    # print(*testSizes)
    # print('sklearn_accuracy')
    # print(*sklearn_accuracy)
    # print('sklearn_precision')
    # print(*sklearn_precision)
    # print('sklearn_recall')
    # print(*sklearn_recall)
    # print('my_accuracy')
    # print(*my_accuracy)
    # print('my_precision')
    # print(*my_precision)
    # print('my_recall')
    # print(*my_recall)
    # print(*my_tp)
    plt.figure(1)
    plt.scatter(testSizes, sklearn_accuracy, label= "sklearn", color='green')
    plt.scatter(testSizes, my_accuracy, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Comparision")
    plt.legend()
    plt.savefig('naive_bayes_acc.png')
    
    plt.figure(2)
    plt.scatter(testSizes, sklearn_precision, label= "sklearn", color='green')
    plt.scatter(testSizes, my_precision, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('Precision')
    plt.title('Precision Comparision')
    plt.legend()
    plt.savefig('naive_bayes_pre.png')

    plt.figure(3)
    plt.scatter(testSizes, sklearn_recall, label= "sklearn", color='green')
    plt.scatter(testSizes, my_recall, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('Recall')
    plt.title('Recall Comparision')
    plt.legend()
    plt.savefig('naive_bayes_rec.png')

    plt.figure(4)
    plt.scatter(testSizes, sklearn_fn, label= "sklearn", color='green')
    plt.scatter(testSizes, my_fn, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('False Negatives')
    plt.title('False Negatives Comparision')
    plt.legend()
    plt.savefig('naive_bayes_fn.png')

    plt.figure(5)
    plt.scatter(testSizes, sklearn_fp, label= "sklearn", color='green')
    plt.scatter(testSizes, my_fp, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('False Positives')
    plt.title('False Positives Comparision')
    plt.legend()
    plt.savefig('naive_bayes_fp.png')

    plt.figure(6)
    plt.scatter(testSizes, sklearn_tp, label= "sklearn", color='green')
    plt.scatter(testSizes, my_tp, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('True Positives')
    plt.title('True Positives Comparision')
    plt.legend()
    plt.savefig('naive_bayes_tp.png')

    plt.figure(7)
    plt.scatter(testSizes, sklearn_tn, label= "sklearn", color='green')
    plt.scatter(testSizes, my_tn, label='my_bayes', color='blue')
    plt.xlabel('Test Sizes')
    plt.ylabel('True Negatives')
    plt.title('True Negatives Comparision')
    plt.legend()
    plt.savefig('naive_bayes_tn.png')

    
    d = {'testSizes':testSizes, 'sklearn_accuracy':sklearn_accuracy, 'my_accuracy':my_accuracy, 'sklearn_precision':sklearn_precision, 'my_precision':my_precision, 'sklearn_recall':sklearn_recall, 'my_recall':my_recall}
    dx = pd.DataFrame(d, columns=['testSizes', 'sklearn_accuracy', 'my_accuracy', 'sklearn_precision', 'my_precision', 'sklearn_recall','my_recall'])
    dx.to_csv('nb_res.csv', sep='\t', encoding='utf-8')
    sys.exit()

if __name__ == '__main__':
    main()