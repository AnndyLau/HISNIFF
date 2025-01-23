import joblib
import os
from sklearn.ensemble import RandomForestRegressor
# 创建随机森林回归模型
#RFR = RandomForestRegressor()
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
#训练数据输入，并转换为特征向量
import ast
from sklearn.naive_bayes import MultinomialNB
import numpy as np
def scale_minmax(col,max,min):
    col = np.array(col)
    max = np.array(max)
    min = np.array(min)
    return (col-min)/(max-min).tolist()
shu_list = ['Mogera','Scapanus','Scaptonyx','Uropsilus','Euroscaptor','Parascaptor','Talpa','18']#,'49']
feature_file = './feature_file/al-5epoch-6+7+8'
concat_lei = 'concat_data'
# ls = [teeth_max,teeth_min,head_max,head_min]
# print(len(ls))
for shu in shu_list:
    #teeth_path = feature_file+'/'+shu+'/teeth_data'
    #head_path = feature_file+'/'+shu+'/head_data'
    teeth_path = './feature_file/al-5epoch-6ceng/'+shu+'/concat_data'
    teeth2_path = './feature_file/al-5epoch-7ceng/'+shu+'/concat_data'
    head_path = './data-al-new/al-5epoch/'+shu+'/concat_data'
    model = MultinomialNB(alpha=1.0)
    #model = SVC()
    #model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5, random_state=42)
    name = str(model)[:-2]
    print('正在读取数据。。。')
    f2=open(os.path.join(teeth_path,'y_train.txt'),mode='r')
    f4=open(os.path.join(teeth_path,'y_test.txt'),mode='r')
    y_train_list=f2.readlines()
    y_test_list=f4.readlines()
    for ls in [y_train_list,y_test_list]:
        for i in range(len(ls)):
            ls[i]=eval(ls[i].strip())
    trian_width = len(y_train_list) 
    x_train=np.zeros(trian_width).tolist()
    y_train=[]
    test_width = len(y_test_list)
    x_test=np.zeros(test_width).tolist()
    y_test=[]
    f2.close()
    f4.close()
    for pathnum,file_path in enumerate([teeth_path,teeth2_path,head_path]):
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
        
        #print(type(x_train_list))
        
        if(pathnum==0):
            print('teeth')
            scaler = MinMaxScaler(feature_range=(0,1))#牙齿占比
            scaler.fit(x_train_list)
            x_train_list_teeth = torch.tensor(scaler.transform(x_train_list))
            x_test_list_teeth = torch.tensor(scaler.transform(x_test_list))
        elif(pathnum==1):
            scaler = MinMaxScaler(feature_range=(0,1))#牙齿占比
            scaler.fit(x_train_list)
            x_train_list_teeth2 = torch.tensor(scaler.transform(x_train_list))
            x_test_list_teeth2 = torch.tensor(scaler.transform(x_test_list))
        else:
            print('head')
            scaler = MinMaxScaler(feature_range=(0,1))
            scaler.fit(x_train_list)
            x_train_list_head = torch.tensor(scaler.transform(x_train_list))
            x_test_list_head = torch.tensor(scaler.transform(x_test_list))

        f1.close()
        f2.close()
        f3.close()
        f4.close()
    #shu = shu+'_2ceng'
    x_train = torch.concat((torch.concat((x_train_list_teeth,x_train_list_teeth2),dim=1),x_train_list_head),dim=1).tolist()
    y_train = y_train_list
    x_test = torch.concat((torch.concat((x_test_list_teeth,x_test_list_teeth2),dim=1),x_test_list_head),dim=1).tolist()
    y_test = y_test_list
    if os.path.exists(os.path.join(feature_file,shu,concat_lei)) is False:
        os.makedirs(os.path.join(feature_file,shu,concat_lei))
    f1=open(os.path.join(feature_file,shu,concat_lei,'x_train.txt'),mode='w')
    f2=open(os.path.join(feature_file,shu,concat_lei,'y_train.txt'),mode='w')
    f3=open(os.path.join(feature_file,shu,concat_lei,'x_test.txt'),mode='w')
    f4=open(os.path.join(feature_file,shu,concat_lei,'y_test.txt'),mode='w')
    f1.writelines('\n'.join(map(str, x_train)))
    f2.writelines('\n'.join(map(str, y_train)))
    f3.writelines('\n'.join(map(str, x_test)))
    f4.writelines('\n'.join(map(str, y_test)))
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    #当算法选用SAMME.R时，基础模型需带有predict_proba()输出概率方法，

    #此时可以进行软投票，否则应使用 algorithm="SAMME"
    # #归一化
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaler.fit(x_train_list)
    # x_train_list = scaler.transform(x_train_list)
    # x_test_list = scaler.transform(x_test_list)
    #直接学习
    print(len(x_train))
    print(len(x_train[0]))
    print(len(y_train))
    print('正在训练。。。')
    model.fit(x_train,y_train)
    print('正在测试。。。')
    train_acc = model.score(x_train,y_train)
    test_acc = model.score(x_test,y_test)
    print('{}属训练集准确率:{}\n测试集准确率:{}'.format(shu,train_acc,test_acc))
    joblib.dump(model, os.path.join(feature_file,shu,concat_lei,name+'_'+str(test_acc)+'.pkl'))
