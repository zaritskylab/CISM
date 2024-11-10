import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score


def get_metrics(df: pd.DataFrame):
    acc = (df['TP'].sum() + df['TN'].sum())/df.shape[0]
    precision = df['TP'].sum()/(df['TP'].sum() + df['FP'].sum())
    recall = df['TP'].sum()/(df['TP'].sum() + df['FN'].sum())
    f1_score = 2 * ((precision*recall)/(precision+recall))
    print(f'recall: {recall}')
    print(f'precision: {precision}')
    print(f'accuracy: {acc}')
    print(f'f1_score: {f1_score}')
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1_score': f1_score}


def get_roc_auc_score(true_y, y_prob, pos_label='TNBC'):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob, pos_label=pos_label)
    return auc(fpr, tpr)


def plot_precision_recall_curve(true_y, y_prob, pos_label='TNBC'):
    """
    plots the roc curve based of the probabilities
    """

    precision, recall, thresholds = precision_recall_curve(true_y, y_prob, pos_label=pos_label)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    print(f"Precision-Recall AUC score: {auc(recall, precision)}")
    return auc(recall, precision)


def plot_roc_curve(true_y, y_prob, pos_label='TNBC'):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob, pos_label=pos_label)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f"FPR-TPR AUC score: {auc(fpr, tpr)}")
    return auc(fpr, tpr)
