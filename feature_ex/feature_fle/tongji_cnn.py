import os
import pandas as pd
from tongji_psbys import tj_bys
d_shu = {}
path_list = ['../data-al-new/weights-al-5epoch']#['./weights-al-5epoch-6ceng','./weights-al-5epoch-7ceng','../data-al-new/weights-al-5epoch','./weights-al-5epoch-6+7','./weights-al-5epoch-6+8','./weights-al-5epoch-7+8','./weights-al-5epoch-6+7+8']
lei_list = ['head','teeth','concat']#,'concat_5:1']
shu_list = ['49','18','Euroscaptor','Mogera','Parascaptor','Scapanus','Scaptonyx','Talpa','Uropsilus']
for shu in shu_list:#18,49,Euroscaptor...
    d_shu[shu]=[]
for path in path_list:
    for shu in shu_list:#18,49,Euroscaptor...
        path2 = os.path.join(path,shu)
        for lei in lei_list:#concat,teeth,head...
            path3 = os.path.join(path2,lei)
            if(path in ['./weights-al-5epoch-6+7','./weights-al-5epoch-6ceng']):
                f = open(os.path.join(path3,'best_acc2.txt'),mode='r')
            else:
                f = open(os.path.join(path3,'best_acc.txt'),mode='r')
            s = f.readlines()
            f.close()
            acc = s[2].split('Acc:')[-1]
            acc = eval(acc)
            record = '{:.2f}%'.format(acc*100)
            d_shu[shu].append(record)
#print(d_shu)
# for shu in d_shu:
#     print(shu,len(d_shu[shu]))
df = pd.DataFrame(d_shu)
df.insert(0,'类型',lei_list)
#df.insert(0,'层数',['6层','7层','8层','6+7','6+8','7+8','6+7+8'])
df2 = tj_bys()
#print(df.T)
#print(df2)
#print(df.T)
num_list = [150,15,23,12,12,11,11,10]
all_acc = []
# acc = df.iloc[0,4]
# print(acc)
for lie in range(3):
    error_pic=0
    for hang in range(2,10):
        acc = df.iloc[lie,hang]
        acc = 1-eval(acc.strip('%'))/100
        error_pic += round(num_list[hang-2]*acc)
        #print(error_pic)
        #print(int(error_pic))
    #error_pic=int(error_pic)
    r_pic = 150-error_pic
    all_acc.append('{:.2f}%'.format(r_pic/1.5))
df.insert(10,'All',all_acc)
#df.insert(10,'朴素贝叶斯',[' ']*7)
df = df.T
# blank_row = pd.DataFrame({col: [''] for col in df.columns})
# # 插入空白行
# df_with_blank_row = df.append(blank_row, ignore_index=True)

df = pd.concat([df,df2])
print(df)
df.to_csv('result-8ceng.csv')