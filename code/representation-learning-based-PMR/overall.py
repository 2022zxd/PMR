import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from dataPrep import *
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import KFold, RepeatedStratifiedKFold

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
        SVM = LogisticRegression(penalty='l2', C=0.5, solver='liblinear', max_iter=1000)
        # SVM = GaussianNB()
        # SVM = RandomForestClassifier(n_estimators=200, random_state=random_state)
        # SVM = SVC(C=1000, probability=True, random_state=random_state)

        df = pd.read_csv(data)
        MR_names, FT_names = find_MR_FTnames(df)

        data = np.asarray(df[FT_names])

        # skf = KFold(n_splits=int(ns), shuffle=False)
        skf = RepeatedStratifiedKFold(n_splits=int(ns), n_repeats=10, random_state=59871)

        totalTP = []
        totalFP = []
        totalTN = []
        totalFN = []
        totalAUC = []
        if len(MR_names) > 1:
            for k in MR_names:
                labels = np.asarray(df[k])
                list_tp = []
                list_fp = []
                list_tn = []
                list_fn = []
                list_auc = []
                for train_index, test_index in skf.split(data, labels):
                    print("TEST:", test_index)
                    x_train, x_test = data[train_index], data[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]
                    model_RWK = SVM.fit(x_train, y_train)

                    prediction = model_RWK.predict_proba(x_test)
                    predic = prediction[:, 1]
                    precision, recall, thresholds = precision_recall_curve(y_test, predic)
                    auc1 = roc_auc_score(y_test, predic)
                    # 找到最佳的阈值
                    f1_scores = 2 * precision * recall / (precision + recall)
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

                    list_tp.append(TP)
                    list_fp.append(FP)
                    list_tn.append(TN)
                    list_fn.append(FN)
                    list_auc.append(auc1)
                    print("TP:" + str(TP))
                    print("FP:" + str(FP))
                    print("TN:" + str(TN))
                    print("FN:" + str(FN))
                if k == 'MR_ADD' :
                    totalTP = list_tp
                    totalFP = list_fp
                    totalTN = list_tn
                    totalFN = list_fn
                    totalAUC = list_auc
                else :
                    for m in range(len(totalTP)):
                        totalTP[m] += list_tp[m]
                    for m in range(len(totalFP)):
                        totalFP[m] += list_fp[m]
                    for m in range(len(totalTN)):
                        totalTN[m] += list_tn[m]
                    for m in range(len(totalFN)):
                        totalFN[m] += list_fn[m]
                    for m in range(len(totalAUC)):
                        totalAUC[m] += list_auc[m]
                print(k)
                print("TP:")
                print(list_tp)
                print("FP:")
                print(list_fp)
                print("TN:")
                print(list_tn)
                print("FN:")
                print(list_fn)
        print("model:")
        print("TP:")
        print(totalTP)
        print("FP:")
        print(totalFP)
        print("TN:")
        print(totalTN)
        print("FN:")
        print(totalFN)
        print("AUC:")
        print(totalAUC)
        acc_total = 0  # 初始化准确率总和
        f1_total = 0
        auc_total = 0

        filename = "/home/zxd/Downloads/RENE-PredictingMetamorphicRelations-main/Phase_III-TrainingTesting/results/total/LR/codeBERT/java/"
        for i in range(len(totalTP)):
            acc = (totalTP[i] + totalTN[i]) / (totalTP[i] + totalTN[i] + totalFP[i] + totalFN[i])  # 计算每个样本的准确率
            acc_total += acc  # 将每个样本的准确率加入准确率总和
            precision = totalTP[i] / (totalTP[i] + totalFP[i])
            recall = totalTP[i] / (totalTP[i] + totalFN[i])
            f1 = 2 * precision * recall / (precision + recall)
            f1_total += f1
            auc = totalAUC[i] / 6
            auc_total += auc

            # 将结果记录到文本文件
            output_file_path = filename + "total_acc.txt"  # 替换为实际的输出文件路径
            with open(output_file_path, 'a') as output_file:
                output_file.write(f"acc {acc}\n")

            # 将结果记录到文本文件
            output_file_path = filename + "total_f.txt"  # 替换为实际的输出文件路径
            with open(output_file_path, 'a') as output_file:
                output_file.write(f"f1 {f1}\n")

            # 将结果记录到文本文件
            output_file_path = filename + "total_auc.txt"  # 替换为实际的输出文件路径
            with open(output_file_path, 'a') as output_file:
                output_file.write(f"auc {auc}\n")

        accuracy = acc_total / len(totalTP)  # 计算样本平均准确率
        fmeasure = f1_total / len(totalTP)
        ave_auc = auc_total / len(totalTP)

        print("Accuracy: {:.3f}".format(accuracy))
        print("Fmeasure: {:.3f}".format(fmeasure))
        print("AUC: {:.3f}".format(ave_auc))

        output_file_path = filename + "total_ave.txt"  # 替换为实际的输出文件路径
        with open(output_file_path, 'a') as output_file:
            output_file.write(f"acc {accuracy}\n")
            output_file.write(f"f1 {fmeasure}\n")
            output_file.write(f"auc {ave_auc}\n")





# python training.py -i "C:\Users\duquet\Documents\GitHub\RENE-PredictingMetamorphicRelations\Phase_II-DataPreparation\Labelled-Dataset\RWK_DS_JK.csv" -ns 10

main()