#################
import numpy as np
from sklearn.metrics import precision_score, classification_report, accuracy_score, recall_score, precision_recall_curve, matthews_corrcoef, f1_score
import sklearn
from scipy.special import softmax
def calculate_precision(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  precision = precision_score(random_true_np,random_pred_np,average= 'binary',zero_division=1)
  return precision



def calculate_recall(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  # precision = precision_score(random_true_np,random_pred_np,average= 'binary')
  recall = recall_score(random_true_np,random_pred_np,average='binary',zero_division=1)
  return recall

def calculate_F1(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  # precision = precision_score(random_true_np,random_pred_np,average= 'binary')
  F1 = sklearn.metrics.f1_score(random_true_np,random_pred_np, average = 'binary',zero_division = 1)
  return F1

def calculate_precision_recall_curve(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  random_pred_np_softmaxed = softmax(random_pred_np, axis=1)
  # precision = precision_score(random_true_np,random_pred_np,average= 'binary')
  precision, recall, _ = precision_recall_curve(random_true_np,random_pred_np_softmaxed[:][1])
  return precision, recall, _

def calculate_precision_recall_by_threshold(random_true,random_pred,threshold):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  random_pred_np = np.swapaxes(random_pred_np,0,1)
  random_pred_np_softmaxed = softmax(random_pred_np,axis=1)
  # random_pred_np_softmaxed = np.swapaxes(random_pred_np_softmaxed,0,1)
  # print(random_true_np.shape, random_pred_np_softmaxed.shape)
  random_pred_np_th = np.where(random_pred_np_softmaxed[:,1] < threshold,0,1)
  precision = precision_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  recall = recall_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  # f1 = (2 * precision * recall)/(precision+recall)
  f1 = f1_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  # precision, recall, _ = precision_recall_curve(random_true_np,random_pred_np[:][1])
  return precision, recall, f1

def save_prediction(random_true,random_pred,threshold):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  random_pred_np = np.swapaxes(random_pred_np,0,1)
  random_pred_np_softmaxed = softmax(random_pred_np,axis=1)
  # random_pred_np_softmaxed = np.swapaxes(random_pred_np_softmaxed,0,1)
  # print(random_true_np.shape, random_pred_np_softmaxed.shape)
  random_pred_np_th = np.where(random_pred_np_softmaxed[:,1] < threshold,0,1)
  # precision = precision_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  # recall = recall_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  # # f1 = (2 * precision * recall)/(precision+recall)
  # f1 = f1_score(random_true_np,random_pred_np_th,average= 'binary',zero_division=1)
  # # precision, recall, _ = precision_recall_curve(random_true_np,random_pred_np[:][1])
  return random_pred_np_th, random_pred_np_softmaxed

def calculate_mattcoeff(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  # precision = precision_score(random_true_np,random_pred_np,average= 'binary')
  matt_coeff = matthews_corrcoef(random_true_np,random_pred_np)
  return matt_coeff

def calculate_classification_report(random_true,random_pred):
  random_true_np = random_true.squeeze().detach().cpu().clone().numpy()
  random_pred_np = random_pred.squeeze().detach().cpu().clone().numpy()
  # precision = precision_score(random_true_np,random_pred_np,average= 'binary')
  matt_coeff = classification_report(random_true_np,random_pred_np)
  return matt_coeff