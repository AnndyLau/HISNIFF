import matplotlib.pyplot as plt
import os
f = open('./datatxt_5split_1/data18-51-zhong-head1level.txt',mode = 'r',encoding='utf-8')
s = f.readlines()
f2 = open('./feature_ex/datatxt_5split_1/data18-51zhong-head.txt',mode= 'r', encoding='utf-8')
s2 = f2.readlines()

f.close()
f2.close()
class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
            'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 
            'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 'Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
            'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
            'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
            'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
            'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
            'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
classname_dict = {}
acc_list = []
classname_dict2 = {}
acc_list2 = []
for i in s:
    classname = i.split('种')[0]
    acc = i.split('种')[1].strip()
    classname_dict[classname]=eval(acc)
for j in s2:
    classname = j.split('种')[0]
    acc = j.split('种')[1].strip()
    classname_dict2[classname]=eval(acc)
for k in class_49:
    acc_list.append(classname_dict[k])
    acc_list2.append(classname_dict2[k])
plt.figure(figsize=(15,5), dpi=80)
plt.plot(range(len(acc_list)),acc_list,label='51',color = 'red')#竖图-width设置线条宽度
plt.plot(range(len(acc_list)),acc_list2,label='18-51',color = 'blue')#竖图-width设置线条宽度
plt.xticks(range(len(class_49)), class_49,  rotation=90)
plt.title('Validation Accuracy')
plt.legend()
if not os.path.exists('./dataimg'):
    os.makedirs('./dataimg')  # 创建路径
plt.savefig('./dataimg/compare.png', dpi=300,bbox_inches='tight')