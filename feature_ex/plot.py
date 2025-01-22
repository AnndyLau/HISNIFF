import matplotlib.pyplot as plt
import os
data_path = './datatxt/data18-49zhong-teeth-od'
f = open(data_path+'.txt',mode = 'r',encoding='utf-8')
s = f.readlines()
s1 = []
for i in s:
    if(i=='\n'):
        continue
    else:
        s1.append(i)
class_18 = ['Condylura', 'Desmana', 'Dymecodon', 'Euroscaptor', 'Galemys', 'Mogera',
            'Neurotrichus', 'Orescaptor', 'Parascalops', 'Parascaptor', 'Scalopus',
            'Scapanulus', 'Scapanus', 'Scaptochirus', 'Scaptonyx', 'Talpa', 'Uropsilus', 'Urotrichus']
class_7 = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 
             'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 
             'Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii',
               'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus',
                 'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1',
                   'Scaptonyx Scaptonyx sp2', 'Scaptonyx Scaptonyx sp4', 'Scaptonyx Scaptonyx sp5', 'Talpa altaica', 'Talpa caeca', 'Talpa davidiana', 'Talpa europaea', 'Talpa romana',
                     'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
acc_list = []
class_dic = {}
if(len(s)==18):
    temp = '属'
    class_list = class_18
elif(len(s)==7):
    temp = '属'
    class_list = class_7
else:
    temp = '种'
    class_list = class_49
for i in s:
    classname = i.split(temp)[0].strip()
    acc = i.split(temp)[1].strip()
    class_dic[classname]=acc
for j in class_list:
    for i in class_dic.keys():
        if(i==j):
            acc_list.append(eval(class_dic[i]))
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08-0.04, 1.02*height, '{:.4f}'.format(height), size=10, family="Times new roman")

plt.figure(figsize=(15,5), dpi=80)
cm = plt.bar(range(len(acc_list)),acc_list,width=0.3)#竖图-width设置线条宽度
if(len(s)==7):
    autolabel(cm)
#plt.plot(range(len(acc_list)),acc_list2,label='47class_afterod',color = 'red')#竖图-width设置线条宽度
##x轴坐标
if not os.path.exists('./dataimg'):
    os.makedirs('./dataimg')  # 创建路径
plt.xticks(range(len(class_list)), class_list,  rotation=90)
#plt.plot(val_acc, label='Validation Accuracy')
plt.title('Validation Accuracy')
#plt.legend()

plt.savefig('./dataimg/'+data_path.split('/')[-1]+'.png', dpi=300,bbox_inches='tight')