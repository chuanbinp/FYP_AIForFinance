#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import confusion_matrix


# In[2]:


def calculate_metrics(results_df, y_class, pred_class, combi_name):
    tn, fp, fn, tp = confusion_matrix(y_class, pred_class).ravel()
    total = tn+fp+fn+tp
    accuracy = (tn+tp)/total
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = (2*precision*recall) / (precision+recall)
    
    results_df = results_df.append({"Experiment": combi_name,
                                    "Accuracy":accuracy, 
                                    "Precision":precision, 
                                    "Recall": recall, 
                                    "F1_score": f1_score}
                                   , ignore_index = True)
    
    return results_df


# In[ ]:


def probability_to_class(proba_pred):
    return [1 if float(pred)>0 else 0 for pred in proba_pred]


# In[ ]:


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

