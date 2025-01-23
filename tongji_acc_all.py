import os
import pandas as pd
from F1 import ff1
class_num = '1'
teeth_weight = './teeth_5zhe_data/teeth_data_'+class_num.split('_')[0]+'/weights-b7-teeth'
head_weight = './head_5zhe_data/head_data_'+class_num.split('_')[0]+'/weights-b7-head'
concat_weight = './feature_ex/feature_file/weights-b7_'+class_num
shu_list = ['51','18','Euroscaptor','Mogera','Parascaptor','Scapanus','Scaptonyx','Talpa','Uropsilus']
zhong_list = [51,18,9,9,3,4,3,7,5]
d = {}
for shu in shu_list:
    d[shu] = [0,0,0]
for shu in shu_list:
    num = zhong_list[shu_list.index(shu)]
    for d_num,t_h in enumerate(['teeth','head','concat']):
        if(d_num == 2):
            t_h_file_name = shu+"/concat_1:1/best_acc.txt"
        else:
            if(shu=='18'):
                t_h_file_name = "EfficientNet_50_"+str(num)+"/best_acc.txt"
            elif(shu=='51'):
                t_h_file_name = "EfficientNet_50_"+str(num)+"/best_acc.txt"
            else:
                t_h_file_name = "EfficientNetb7-"+t_h+"_50_"+str(num)+"_"+shu+"/best_acc.txt"
        file_path = os.path.join(eval(t_h+"_weight"),t_h_file_name)
        f = open(file_path,mode='r')
        s = f.readlines()
        f.close()
        acc = eval(s[-1].split(':')[-1].strip())
        d[shu][d_num] = '{:.2f}%'.format(acc*100)
        #print(acc)
l1 = []
for lei in ['teeth','head','concat']:
    l1.append(ff1(lei,class_num))
for i,score in enumerate(['Accuracy','Precision','Recall','F1 Score']):
    teeth = "{:.2f}%".format(l1[0][i]*100)
    head = "{:.2f}%".format(l1[1][i]*100)
    concat = "{:.2f}%".format(l1[2][i]*100)
    d[score] = [teeth,head,concat]

df = pd.DataFrame(d).T
df.columns = ['teeth','head','concat']
df.to_excel('./feature_ex/datatxt_b7_'+class_num+'/acc_all_'+class_num+'.xlsx')
print(df)