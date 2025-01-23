from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score
import numpy as np
import pandas as pd
from collections import defaultdict
# 假设你有一个名为 confusion_matrix 的 51x51 混淆矩阵
# 替换这里的数据为你的实际混淆矩阵数据
#ata_path = './feature_ex/datatxt_r_p/data18-51-combine'
def calculate_recall(original_labels, predicted_labels, num_classes=51):
    # 初始化 True Positives (TP) 和 False Negatives (FN) 的字典
    tp_fn_dict = defaultdict(lambda: {'TP': 0, 'FN': 0})
    
    # 统计 TP 和 FN
    for original_label, predicted_label in zip(original_labels, predicted_labels):
        if original_label == predicted_label:
            tp_fn_dict[original_label]['TP'] += 1
        else:
            tp_fn_dict[original_label]['FN'] += 1
    
    # 计算每个类别的召回率
    recalls = []
    for class_index in range(num_classes):
        tp = tp_fn_dict[class_index]['TP']
        fn = tp_fn_dict[class_index]['FN']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 避免除以0
        recalls.append(recall)
    
    # 计算平均召回率
    avg_recall = np.mean(recalls)
    return avg_recall
def ff1(lei,class_num):
    if(lei =='concat'):
        data_path = './feature_ex/datatxt_b7_'+class_num+'/data18-51-combine'
    else:
        data_path = './feature_ex/datatxt_b7_'+class_num+'/data18-51zhong-'+lei
    model = 'weighted'
    f1 = open(data_path+'-y_ture.txt',mode = 'r',encoding='utf-8')
    y_ture = f1.readlines()
    f2 = open(data_path+'-y_pred.txt',mode = 'r',encoding='utf-8')
    y_pred = f2.readlines()
    f1.close()
    f2.close()
    for i in range(len(y_ture)):
        y_ture[i] = int(y_ture[i])
        y_pred[i] = int(y_pred[i])
    y_true = np.array(y_ture)
    y_pred = np.array(y_pred)
    # from sklearn.metrics import classification_report
    # print(classification_report(y_true=y_true, y_pred=y_pred))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=model)
    recall = calculate_recall(y_true, y_pred)
    #recall = recall_score(y_true, y_pred,average=model)

    # 计算 F1 分数
    ff = f1_score(y_true, y_pred, average=model)

    # 打印结果
    print(f'Accuracy:{acc*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1 Score: {ff*100:.2f}%')
    return [acc,precision,recall,ff]
def ff2(lei,class_num):
    if(lei =='concat'):
        data_path = './datatxt_b7_'+class_num+'/data51-combine'
    else:
        data_path = './datatxt_b7_'+class_num+'/data18-51-zhong-'+lei+'1level'
    model = 'weighted'
    f1 = open(data_path+'-y_ture.txt',mode = 'r',encoding='utf-8')
    y_ture = f1.readlines()
    f2 = open(data_path+'-y_pred.txt',mode = 'r',encoding='utf-8')
    y_pred = f2.readlines()
    f1.close()
    f2.close()
    for i in range(len(y_ture)):
        y_ture[i] = int(y_ture[i])
        y_pred[i] = int(y_pred[i])
    y_true = np.array(y_ture)
    y_pred = np.array(y_pred)
    # from sklearn.metrics import classification_report
    # print(classification_report(y_true=y_true, y_pred=y_pred))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=model)
    recall = calculate_recall(y_true, y_pred)
    #recall = recall_score(y_true, y_pred,average="macro")
    # 计算 F1 分数
    ff = f1_score(y_true, y_pred, average=model)

    # 打印结果
    print(f'Accuracy:{acc*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1 Score: {ff*100:.2f}%')
    return [acc,precision,recall,ff]
# d={}
# l1=[]
# class_num='1'
# for lei in ['teeth','head','concat']:
#     l1.append(ff2(lei,class_num))
# for i,score in enumerate(['Accuracy','Precision','Recall','F1 Score']):
#     teeth = "{:.2f}%".format(l1[0][i]*100)
#     head = "{:.2f}%".format(l1[1][i]*100)
#     concat = "{:.2f}%".format(l1[2][i]*100)
#     d[score] = [teeth,head,concat]

# df = pd.DataFrame(d).T
# df.columns = ['teeth','head','concat']
# df.to_excel('./datatxt_b7_'+class_num+'/acc_51_'+class_num+'.xlsx')