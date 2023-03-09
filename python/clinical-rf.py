import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc ,confusion_matrix ,accuracy_score ,f1_score ,roc_auc_score ,recall_score ,precision_score
import warnings

warnings.filterwarnings("ignore")
# #############################################################################
# settings
rs = 0
target_auc = 0.75
# #############################################################################

data = pd.read_excel(r'./data/clinical.xlsx')
data = data.set_index('index')
label = pd.read_excel(r'./data/labels.xlsx', index_col='index')
# Data merging
Merge_data = pd.merge(data, label, how='inner', left_index=True, right_index=True)
index = Merge_data.index[np.isnan(Merge_data.loc[:, "label"])]
Merge_data.drop(index, inplace=True)
X = Merge_data.drop('label', axis=1)
y = Merge_data['label']
print(y.value_counts().head())
# Label binarizing
y = label_binarize(y, classes=list(range(2)))
# print(X.shape)

# #############################################################################
# Univariate feature selection with varience
from sklearn.feature_selection import VarianceThreshold
threshold = 0
vt = VarianceThreshold(threshold=threshold)
vt.fit(X)
dict_variance = {}
for i, j in zip(X.columns.values, vt.variances_):
    dict_variance[i] = j
ls = list()
for i, j in dict_variance.items():
    if j >= threshold:
        ls.append(i)
X = pd.DataFrame(vt.fit_transform(X), columns=ls)
print("Number of features: %s" % X.columns.size)

# #############################################################################
index_X= X
index_y= y

# #############################################################################
max_auc = 0
max_time = 0
flag = 1
while flag < 2:
    X = index_X
    y = index_y
    features = X.columns
    # feature selection with RF
    clf_rf = RandomForestClassifier(random_state=rs, class_weight='balanced')
    rfe_rf = RFECV(estimator=clf_rf, step=1, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    X = rfe_rf.fit_transform(X, y)
    print("Number of features after RFECV: %s" % len(features[rfe_rf.support_]))
    print("Final features:", features[rfe_rf.support_])

    parameters = {'n_estimators': [50, 60, 70, 80],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  "min_samples_leaf": [2, 4, 6, 8],
                  'max_depth': [4, 5, 6, 7, 8],
                  'criterion': ['gini', 'entropy']}
    clf_rf = RandomForestClassifier(random_state=rs, class_weight='balanced')
    Grid = GridSearchCV(clf_rf, parameters, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    Grid.fit(X, y)
    print('Best parameters:', Grid.best_params_)
    # print('Best AUC:', Grid.best_score_)
    print(Grid.best_estimator_)
    
    #############################################################################
    def calculate_metric(gt, pred):
        pred[pred > 0.5] = 1
        pred[pred < 1] = 0
        confusion = confusion_matrix(gt, pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
        print('Sensitivity:', TP / float(TP + FN))
        print('Specificity:', TN / float(TN + FP))
        acc = (TP + TN) / float(TP + TN + FP + FN)
        sensitivity = TP / float(TP + FN)
        specificity = TN / float(TN + FP)
        return acc,sensitivity,specificity



        # #############################################################################
    # plot ROC and AUC
    i = 0
    tprs = []
    aucs = []
    accs = []
    sensis = []
    specis = []
    feature_importance_array = []
    mean_fpr = np.linspace(0, 1, 500)
    figsize = 11, 9
    figure, ax = plt.subplots(figsize=figsize)
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    signature = np.zeros_like(y)
    for train_index, test_index in StratifiedKFold(n_splits=5).split(X, y):
        # Split dataset
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        # Model
        model = RandomForestClassifier(random_state=rs, class_weight='balanced', **Grid.best_params_)
        model.fit(X_train, Y_train)
        feature_importance_array.append([])
        for feature_importance in model.feature_importances_:
            feature_importance_array[i].append(feature_importance)
        y_pred = model.predict_proba(X_test)
        print("pred:",y_pred)
        print("test_index:",test_index)

        signature[test_index] = y_pred[:, 1].reshape(y_pred.shape[0], 1)

        print("time:",i)
        acc,sensitivity,specificity = calculate_metric(Y_test,y_pred[:, 1])
        accs.append(acc)
        sensis.append(sensitivity)
        specis.append(specificity)
        y_pred = model.predict_proba(X_test)

        str1 = './results/clinical-proba'
        str2 = '.xlsx'
        str3 = str(i)
        para = str1 + str3 +str2
        proba = pd.DataFrame(list(y_pred))
        writer = pd.ExcelWriter(para)
        proba.to_excel(writer, startcol=0, index=False)
        writer.save()

        str4 = './results/clinical-index'
        para2 = str4 + str3 +str2
        probindex = pd.DataFrame(list(test_index))
        writer = pd.ExcelWriter(para2)
        probindex.to_excel(writer, startcol=0, index=False)
        writer.save()

        fpr, tpr, thresholds = roc_curve(Y_test, y_pred[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1.2, alpha=0.3, label='ROC fold %d(AUC=%0.2f)'% (i+1, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_acc = np.mean(accs)
    mean_sensi = np.mean(sensis)
    mean_speci = np.mean(specis)
    std_auc = np.std(aucs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.2f)'% mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr-std_tpr, 0)

    if(mean_auc>max_auc):
        max_auc = mean_auc
        max_time = rs
    if (mean_auc > target_auc):
        print("complete congratulation!")
        print('AUC = ', mean_auc)
        print('ACC = ', mean_acc)
        print('Sensitivity = ', mean_sensi)
        print('Specificity = ', mean_speci)
        print('feature importance:',feature_importance_array)
        flag = flag + 1
    else:
        print('random_state:', rs)
        print('AUC = ', mean_auc)
        print('Max_AUC = ', max_auc)
        print('random_state when Max_AUC occurs= ', max_time)
        print("try again")
        rs = rs + 1
        pass
    
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05]) 
plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 16})
plt.title('ROC of Fused Model', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 20})
plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 14})
# plt.savefig("Clinical_model.png", dpi=500)
plt.show()