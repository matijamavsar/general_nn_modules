import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

def evaluate_classifier(y_true,y_pred,labels=None,sample_weight=None,normalize=None):
    # Get confusion matrix
    calc_cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=normalize)
    
    fig = plt.figure(figsize=(10,10))
    plt.matshow(calc_cm,fignum=0)
    # Return plot of confusion matrix
    
    return calc_cm
