##coding:utf-8
import os.path
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import rcParams
#plt.style.use('ggplot')
data_path = '/home/cz/data/final_data/datatxt_b7_1/data18-51-zhong-teeth1level'
#data_path = './datatxt_5split_1/data18-51-zhong-head1level-shu'
f = open(data_path+'-matrix.txt',mode = 'r',encoding='utf-8')
class_18 = ['Condylura', 'Desmana', 'Dymecodon', 'Euroscaptor', 'Galemys', 'Mogera',
            'Neurotrichus', 'Orescaptor', 'Parascalops', 'Parascaptor', 'Scalopus',
            'Scapanulus', 'Scapanus', 'Scaptochirus', 'Scaptonyx', 'Talpa', 'Uropsilus', 'Urotrichus']
# 定义混淆矩阵数据
class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
            'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 
            'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 'Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
            'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
            'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
            'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
            'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
            'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
labels = class_49
s=f.read()
print(s)
confusion_matrix = np.array(eval(s))#np.zeros((47,47))
print(type(confusion_matrix))
print(confusion_matrix)
# 计算每个类别的准确率
class_accuracy = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]

# 绘制混淆矩阵图像
# 要想改变颜色，修改cmap参数，红色：plt.cm.Reds
plt.figure(figsize=(15,15), dpi=80)
plt.imshow(class_accuracy, cmap=plt.cm.Oranges)

# 添加网格
plt.grid(False)
#plt.colorbar()
# labels表示你不同类别的代号，这里有5个类别

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize=10,rotation=90)
plt.yticks(tick_marks, labels, fontsize=10)

# 在每个小格子中显示相应的数值和准确率
for i, j in itertools.product(range(class_accuracy.shape[0]), range(class_accuracy.shape[1])):
    # 显示数值
    value = confusion_matrix[i, j]
    plt.text(j, i, int(value), verticalalignment='center',horizontalalignment="center", color="white" if class_accuracy[i, j] > 0.5 else "black")

    # 显示准确率
    #acc = class_accuracy[i, j] * 100
    #plt.text(j, i + 0.3, f"{acc:.2f}%", horizontalalignment="center", color="black")

# 添加x和y轴标签
img_path  = '/'.join(data_path.split('/')[:-1])+'/'
# if not os.path.exists('./dataimg'):
#     os.makedirs('./dataimg')  # 创建路径
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")
plt.savefig(img_path+data_path.split('/')[-1]+'-matrix.png', dpi=300,bbox_inches='tight')
# 显示图像
#plt.show()

