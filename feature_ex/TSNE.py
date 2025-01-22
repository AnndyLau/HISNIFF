import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
# 创建随机森林回归模型
#RFR = RandomForestRegressor()
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
#训练数据输入，并转换为特征向量
import ast
from sklearn.naive_bayes import MultinomialNB,GaussianNB
shu_list = ['18']#['Mogera','Parascaptor','Scaptonyx','Uropsilus','Euroscaptor','Scapanus','Talpa']
server_name = 'b7_1'
for shu in shu_list:
    for lei in ['concat_1:1']:
        x_test_list=[]
        y_test_list=[]
        file_path = './feature_file/'+server_name+'/'+shu+'/'+lei+'_data'
        print('正在读取数据。。。')

        f1=open(os.path.join(file_path,'x_train.txt'),mode='r')
        f2=open(os.path.join(file_path,'y_train.txt'),mode='r')
        f3=open(os.path.join(file_path,'x_test.txt'),mode='r')
        f4=open(os.path.join(file_path,'y_test.txt'),mode='r')
        x_train_list=f1.readlines()
        #print('zhe')
        y_train_list=f2.readlines()
        x_test_list=f3.readlines()
        #print(len(x_test_list[0]))
        y_test_list=f4.readlines()
        for ls in [x_train_list,y_train_list,x_test_list,y_test_list]:
            for i in range(len(ls)):
                ls[i]=eval(ls[i].strip())
        f1.close()
        f2.close()
        f3.close()
        f4.close()
    # 生成样本数据
        x_train_list = np.array(x_train_list)
        y_train_list = np.array(y_train_list)
        x_test_list = np.array(x_test_list)
        y_test_list = np.array(y_test_list)
        if(lei in ['head','teeth']):
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler.fit(x_train_list)
            x_train_list = scaler.transform(x_train_list)
            x_test_list = scaler.transform(x_test_list)
        else:
            lei = 'fusion'
        X = np.concatenate((x_train_list, x_test_list), axis=0)
        y = np.concatenate((y_train_list, y_test_list), axis=0)
        # 创建t-SNE模型并拟合数据
        tsne = TSNE(n_components=2, random_state=42)  # 指定降维后的维度为2
        X_embedded = tsne.fit_transform(X)

        # 定义不同类别的颜色
        color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        marker_list = ['o','^','*','s', 'D']
        # 可视化降维后的数据，根据类别进行标记
        plt.figure(figsize=(8, 6))
        for i in range(len(np.unique(y))):
            plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1],c=color_list[i%7],marker=marker_list[i%5],label=f'Class {i}',alpha=0.7,edgecolors='black')

        plt.title('Genus')
        plt.xlabel('T-SNE Dimension 1')
        plt.ylabel('T-SNE Dimension 2')
        #plt.legend()
        plt.savefig('./shu_tu/'+shu+lei+'.png', dpi=300,bbox_inches='tight')
        plt.show()
        plt.close()
