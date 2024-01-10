import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from dataPrep import *
from sklearn.svm import SVC
from sklearn.metrics import *
from scipy import interp
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})


def saveTrainTesting(mainPath, x, y, mr, j, ns, t):
    if t == 'tr':
        pkl_name_train_x = mainPath + '/' + 'TrainingX_' + mr + '_' + j + '_' + ns + 'folds.pkl'
        with open(pkl_name_train_x, 'wb') as file:
            pickle.dump(x, file)

        pkl_name_train_y = mainPath + '/' + 'TrainingY_' + mr + '_' + j + '_' + ns + 'folds.pkl'

        with open(pkl_name_train_y, 'wb') as file:
            pickle.dump(y, file)

    if t == 'ts':
        pkl_name_test_x = mainPath + '/' + 'TestingX_' + mr + '_' + j + '_' + ns + 'folds.pkl'

        with open(str(pkl_name_test_x), 'wb') as file:
            pickle.dump(x, file)

        pkl_name_test_y = mainPath + '/' + 'TestingY_' + mr + '_' + j + '_' + ns + 'folds.pkl'
        with open(pkl_name_test_y, 'wb') as file:
            pickle.dump(y, file)


def saveModel(mainPath, model, mr, j, ns):
    pkl_name_train_x = mainPath + '/' + 'model_' + mr + '_' + j + '_' + ns + 'folds.pkl'

    with open(pkl_name_train_x, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    import click


    @click.command()
    @click.option('-i', '--file', 'data', help='Path of labelled Dataset')
    @click.option('-ns', '--nSplits', 'ns', help='Number of splits')
    def main(data, ns):

        mainPath_name = data.split('/')[-1]
        mainPath_name = mainPath_name.split('.')[0]
        mainPath = storage_file(mainPath_name)

        random_state = np.random.RandomState(0)
        # SVM = LogisticRegression(penalty='l2', C=0.5, solver='liblinear', max_iter=1000)
        # SVM = GaussianNB()
        # SVM = RandomForestClassifier(n_estimators=200, random_state=random_state)
        SVM = SVC(C=1000, probability=True, random_state=random_state)

        results_folder = "/home/zxd/Downloads/RENE-PredictingMetamorphicRelations-main/Phase_III-TrainingTesting/results/3/SVM/NF-PF/python"  # 设置结果文件夹的路径
        os.makedirs(results_folder, exist_ok=True)  # 创建结果文件夹，如果它不存在 codeBERT NF-PF GraphCodeBERT infercode HiT UniXcoder ncc Mocktail ast2vec
        number = '0'

        skf = RepeatedStratifiedKFold(n_splits=int(ns), n_repeats=10, random_state=3)
        # skf = RepeatedStratifiedKFold(n_splits=int(ns), n_repeats=10,random_state=59871)
        # skf = KFold(n_splits=int(ns), shuffle=True)
        df = pd.read_csv(data)
        MR_names, FT_names = find_MR_FTnames(df)

        data = np.asarray(df[FT_names])
        # print(data)

        if len(MR_names) > 1:
            for i in MR_names:
                mainPath2 = createDir(mainPath_name, i)
                nameModel = 'Models_' + i
                modelPath = createDir(mainPath_name, nameModel)
                labels = np.asarray(df[i])
                j = 0
                list_acc = []
                list_f1 = []
                list_auc = []

                for train_index, test_index in skf.split(data, labels):
                    x_train, x_test = data[train_index], data[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]

                    print(f"{i}+测试集索引:", test_index)

                    # print("测试集data:", data[test_index])
                    # print("测试集labels:", labels[test_index])

                    j = j + 1
                    saveTrainTesting(mainPath2, x_train, y_train, i, str(j), ns, t='tr')
                    saveTrainTesting(mainPath2, x_test, y_test, i, str(j), ns, t='ts')
                    model_RWK = SVM.fit(x_train, y_train)
                    saveModel(modelPath, model_RWK, i, str(j), ns)

                    prediction = model_RWK.predict_proba(x_test)
                    predic = prediction[:, 1]

                    precision, recall, thresholds = precision_recall_curve(y_test, predic)
                    print(precision, recall, thresholds)
                    auc1 = roc_auc_score(y_test, predic)
                    # 找到最佳的阈值
                    f1_scores = 2 * precision * recall / (precision + recall)
                    # print('precision:', precision, 'recall:',recall, 'thresholds:',thresholds, 'f1_scores:',f1_scores)
                    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
                    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                    # 阈值
                    best_threshold = thresholds[best_f1_score_index]

                    # 根据最佳阈值计算TP、TN、FP和FN
                    y_pred = np.where(predic >= best_threshold, 1, 0)
                    TP = np.sum((y_test == 1) & (y_pred == 1))
                    TN = np.sum((y_test == 0) & (y_pred == 0))
                    FP = np.sum((y_test == 0) & (y_pred == 1))
                    FN = np.sum((y_test == 1) & (y_pred == 0))

                    # 根据TP、TN、FP和FN计算precision、recall、accuracy等指标
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    accuracy = (TP + TN) / len(y_test)
                    f1_scores = 2 * precision * recall / (precision + recall)



                    # # 计算混淆矩阵
                    # confusion_mat = confusion_matrix(y_test, (predic > 0.1).astype(int))
                    # # 提取TP、TN、FP、FN
                    # TP = confusion_mat[1, 1]
                    # TN = confusion_mat[0, 0]
                    # FP = confusion_mat[0, 1]
                    # FN = confusion_mat[1, 0]
                    #
                    # # 输出每个折叠的TP、TN、FP、FN
                    # print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
                    #
                    # precision = TP / (TP + FP)
                    # recall = TP / (TP + FN)
                    # accuracy = (TP + TN) / len(y_test)
                    # f1_scores = 2 * precision * recall / (precision + recall)



                    list_acc.append(accuracy)
                    list_f1.append(f1_scores)
                    list_auc.append(auc1)

                    result_file_path1 = os.path.join(results_folder, f'test_index_{number}.txt')
                    with open(result_file_path1, 'a') as file:
                        file.write(f"{i}_测试集索引: " + str(test_index) + "\n")

                mean_acc = np.mean(list_acc)
                mean_f1 = np.mean(list_f1)
                mean_auc = np.mean(list_auc)

                print(i)
                print("Accuracy:",list_acc)
                print("F1 score:",list_f1)
                print("AUC: ",list_auc)
                print("Mean Accuracy: {:.3f}".format(mean_acc))
                print("Mean F1 score: {:.3f}".format(mean_f1))
                print("Mean AUC: {:.3f}".format(mean_auc))

                # 定义三个文件名
                file_names = [f"ave_accuracy_{number}.txt", f"ave_f1_score_{number}.txt", f"ave_auc_{number}.txt"]

                # 目标值列表
                mean_values = [mean_acc, mean_f1, mean_auc]

                # 遍历每个文件名和目标值，将它们写入相应的文件
                for file_name, mean_value in zip(file_names, mean_values):
                    result_file_path = os.path.join(results_folder, file_name)
                    with open(result_file_path, 'a') as file:
                        file.write(f"{i} {mean_value:.3f}\n")


                # 定义要保存的文件名和数据
                file_data = [
                    (f"all_accuracy_{number}.txt", list_acc),
                    (f"all_f1_score_{number}.txt", list_f1),
                    (f"all_auc_{number}.txt", list_auc)
                ]

                # 遍历文件名和数据
                for file_name, data_list in file_data:
                    result_file_path = os.path.join(results_folder, file_name)
                    with open(result_file_path, 'a') as file:
                        for K, value in enumerate(data_list):
                            file.write(f"{i} Entry {K}: {value:.3f}\n")

main()