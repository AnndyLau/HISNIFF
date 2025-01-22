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
shu_list = ['51']#['18','Mogera','Parascaptor','Scaptonyx','Uropsilus','Euroscaptor','Scapanus','Talpa']
server_name = 'b7_1'
#model = MultinomialNB(alpha=1.0)
model  = GaussianNB()
#model = SVC()
#model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5, random_state=42)
model_list = [MultinomialNB(alpha=1.0),GaussianNB(),SVC(),DecisionTreeClassifier(),LogisticRegression(),RandomForestClassifier(n_estimators=200,random_state=42)]
for model in model_list:
    name = str(model)[:-2]
    f = open('./feature_file/weights-'+server_name+'/'+name+'.txt',mode='w')
    for shu in shu_list:
        file_path = './feature_file/'+server_name+'/'+shu+'/head_data'
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
        #直接学习
        print(len(x_train_list))
        print(len(x_train_list[0]))
        print('正在训练。。。')
        model.fit(x_train_list,y_train_list)
        #model = joblib.load("./feature_file/Mogera/head_data/MultinomialNB_0.8695652173913043.pkl")
        print('正在测试。。。')
        train_acc = model.score(x_train_list,y_train_list)
        test_acc = model.score(x_test_list,y_test_list)
        f.write('{}属训练集准确率:{}\n测试集准确率:{}\n'.format(shu,train_acc,test_acc))
        print('{}属训练集准确率:{}\n测试集准确率:{}'.format(shu,train_acc,test_acc))
        joblib.dump(model, os.path.join(file_path,name+'_'+str(test_acc)+'.pkl'))
    f.close()
