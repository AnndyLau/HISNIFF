import os
import pandas as pd
def tj_bys():
    d_shu = {}
    path_list = ['../data-al-new/al-5epoch']#['./al-5epoch-6ceng','./al-5epoch-7ceng','../data-al-new/al-5epoch','./al-5epoch-6+7','./al-5epoch-6+8','./al-5epoch-7+8','./al-5epoch-6+7+8']
    lei_list = ['concat']#['head','teeth','concat']
    shu_list = ['18','Euroscaptor','Mogera','Parascaptor','Scapanus','Scaptonyx','Talpa','Uropsilus']
    for shu in shu_list:#18,49,Euroscaptor...
        d_shu[shu]=[]
    for path in path_list:
        for shu in shu_list:#18,49,Euroscaptor...
            path2 = os.path.join(path,shu)
            for lei in lei_list:#concat,teeth,head...
                path3 = os.path.join(path2,lei+'_data')
                for name in ['MultinomialNB','SVC','DecisionTreeClassifier','KNeighborsClassifier','GaussianNB']:
                    for file in os.listdir(path3):#x_test.txt,x_train.txt...
                        if(file.split('_')[0]==name):
                            acc = eval(file.split('_')[-1][:-4])
                    record = '{:.2f}%'.format(acc*100)
                    d_shu[shu].append(record)
    #print(d_shu)
    # for shu in d_shu:
    #     print(shu,len(d_shu[shu]))
    df = pd.DataFrame(d_shu)
    df.insert(0,'类型/分类器',['MultinomialNB','SVC','DecisionTreeClassifier','KNeighborsClassifier','GaussianNB'])
    print(df.T)
    #df.insert(0,'层数',['6层','7层','8层','6+7','6+8','7+8','6+7+8'])
    num_list = [150,15,23,12,12,11,11,10]
    all_acc = []
    # acc = df.iloc[0,4]
    # print(acc)
    for lie in range(5):
        error_pic=0
        for hang in range(1,9):
            acc = df.iloc[lie,hang]
            print(acc)
            acc = 1-eval(acc.strip('%'))/100
            error_pic += round(num_list[hang-1]*acc)
            #print(error_pic)
            #print(int(error_pic))
        #error_pic=int(error_pic)
        r_pic = 150-error_pic
        all_acc.append('{:.2f}%'.format(r_pic/1.5))
    df.insert(9,'All',all_acc)
    #print(df.T)
    return df.T
df = tj_bys()
print(df)
df.to_csv('result-sklearn.csv')