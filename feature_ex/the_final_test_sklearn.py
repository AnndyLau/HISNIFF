import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision import transforms
import os
from feature_file.outfeature import out_feature,out_feature_18,image_process,guiyihua,guiyihua2
from PIL import Image
from feature_file.liner_model import SimpleNet2,SimpleNet
from sklearn.preprocessing import MinMaxScaler
from feature_file.GYH import scaler_tool
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
#训练数据输入，并转换为特征向量
import ast
from skopt import BayesSearchCV
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
class_18 = ['Condylura', 'Desmana', 'Dymecodon', 'Euroscaptor', 'Galemys', 'Mogera', 
            'Neurotrichus', 'Orescaptor', 'Parascalops', 'Parascaptor', 'Scalopus', 
            'Scapanulus', 'Scapanus', 'Scaptochirus', 'Scaptonyx', 'Talpa', 'Uropsilus', 'Urotrichus']
class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 
             'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 
             'Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii',
               'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus',
                 'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1',
                   'Scaptonyx Scaptonyx sp2', 'Scaptonyx Scaptonyx sp4', 'Scaptonyx Scaptonyx sp5', 'Talpa altaica', 'Talpa caeca', 'Talpa davidiana', 'Talpa europaea', 'Talpa romana',
                     'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
class_1 = ['Talpa altaica', 'Talpa caeca', 'Talpa davidiana', 'Talpa europaea', 'Talpa romana']
#l_class_18 = list(class_18.items())
model_list = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
num_list = [5,5,4,4,3,9,8]
shu_tool = []
print('正在读取归一化工具')
for shu in model_list:
    teeth_tool = joblib.load('./data-al-new/scaler_dir_new/'+shu+'_teeth.pkl')
    head_tool = joblib.load('./data-al-new/scaler_dir_new/'+shu+'_head.pkl')
    shu_tool.append([teeth_tool,head_tool])
teeth_tool = joblib.load('./data-al-new/scaler_dir_new/18_teeth.pkl')
head_tool = joblib.load('./data-al-new/scaler_dir_new/18_head.pkl')
shu_tool.append([teeth_tool,head_tool])
zhong_model_list = [['Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes'],
                    ['Talpa altaica', 'Talpa caeca', 'Talpa davidiana', 'Talpa europaea', 'Talpa romana'],
                    ['Scaptonyx Scaptonyx sp1', 'Scaptonyx Scaptonyx sp2', 'Scaptonyx Scaptonyx sp4', 'Scaptonyx Scaptonyx sp5'],
                    ['Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii'],
                    ['Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    ['Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura'],
                    ['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
input_size=480
transform = torchvision.transforms.Compose(
    [
            transforms.Resize((input_size,input_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
head_path = './head_al_49_1/test'
teeth_path = './data-al-new/teeth_al_49_new/test'
#teeth_path = './data-al-new/teeth_al_49_new/test_od'
process = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    #nn.MaxPool2d(kernel_size=3, stride=3,padding=1),
    #nn.MaxPool2d(kernel_size=3, stride=3),
    nn.Flatten(start_dim=1),
    #nn.Dropout(0.5),
    #nn.Linear(2048,2048),
    nn.ReLU(inplace=True),
)
process.to(device)
# net_18 = torch.load('./data-al-new/weights-al-5epoch/18/concat/best_model.pkl',map_location=device)
# net_18.to(device)
# net_18.eval()
net_18 = joblib.load('/home/cz/data/feature-mouse/data_al/data-al-new/al-5epoch/18/concat_data/GaussianNB_0.8933333333333333.pkl')
r_num = 0
shu_r_num =0
img_num = 0
txt_path = './datatxt/'
server_name = 'data18-49zhong-combine-sk'
matrix = np.zeros((49,49))
if os.path.exists(txt_path) is False:
    os.makedirs(txt_path)
f= open(txt_path+server_name+'.txt',mode = 'w')
f2 = open(txt_path+server_name+'-matrix.txt', mode = 'w')
f3 = open(txt_path+server_name+'-error.txt',mode='w')
print(shu_tool)
for shu in class_49:
    zhong_acc = 0
    zhong_num = 0
    head_imagedir_path = os.path.join(head_path,shu)
    teeth_imagedir_path = os.path.join(teeth_path,shu)
    for img in os.listdir(head_imagedir_path):#头骨图片名与牙齿图片名一样
        img_num+=1
        zhong_num+=1
        #print('正在测试第{}张图片'.format(img_num))
        head_img_path = os.path.join(head_imagedir_path,img)
        teeth_img_path = os.path.join(teeth_imagedir_path,img)
        head_image = image_process(head_img_path,transform,input_size)
        teeth_image = image_process(teeth_img_path,transform,input_size)
        head_feature_shu = out_feature_18(teeth_image,head_image,'head')
        teeth_feature_shu = out_feature_18(teeth_image,head_image,'teeth')
        with torch.no_grad():
            head_feature_shu = process(head_feature_shu)
            teeth_feature_shu = process(teeth_feature_shu)
            concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[-1][0],shu_tool[-1][1])
            shu_output = net_18.predict(concat_feature_shu.cpu())
            shu_result = class_18[int(shu_output[0])]
            r_shu = shu.split()[0]
            if(shu_result==r_shu):
                shu_r_num+=1
                #continue
                #print('判断正确')
                if(shu_result not in model_list):
                    r_num+=1
                    hang = class_49.index(shu)
                    lie = class_49.index(shu)
                    matrix[hang][lie]+=1
                    zhong_acc+=1
                    #break
                    continue
            else:
                print('{}---->{}'.format(img,shu_result))
                f3.write('{}---->{}\n'.format(img,shu_result))
                matrix[class_49.index(shu)][48]+=1
                #break
                continue
        shu_id = model_list.index(shu_result)
        num_class = num_list[shu_id]
        head_feature_shu = out_feature(teeth_image,head_image,num_class,'head',shu_result)
        teeth_feature_shu = out_feature(teeth_image,head_image,num_class,'teeth',shu_result)
        head_feature_shu = process(head_feature_shu)
        teeth_feature_shu = process(teeth_feature_shu)
        concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[shu_id][0],shu_tool[shu_id][1])
        # net_49 = torch.load(os.path.join('./data-al-new/weights-al-5epoch',shu_result,'concat','best_model.pkl'),map_location=device)
        # net_49.eval()
        for name in os.listdir(os.path.join('/home/cz/data/feature-mouse/data_al/data-al-new/al-5epoch',shu_result,'concat_data')):
            if(name.split('_')[0]=='GaussianNB'):#['MultinomialNB','SVC','DecisionTreeClassifier','KNeighborsClassifier','GaussianNB']
                break
        net_49 = joblib.load(os.path.join('/home/cz/data/feature-mouse/data_al/data-al-new/al-5epoch',shu_result,'concat_data',name))
        zhong_output = net_49.predict(concat_feature_shu.cpu())
        #zhong_result = zhong_output.argmax(1)
        #zhong_result = str(zhong_result).split('[')[1].split(']')[0]
        zhong_result = zhong_model_list[shu_id][int(zhong_output[0])]
        r_zhong = shu
        hang = class_49.index(r_zhong)
        lie = class_49.index(zhong_result)
        if(zhong_result==r_zhong):
            #print('种判断正确')
            r_num+=1
            zhong_acc+=1
            matrix[hang][lie]+=1
            continue
        else:
            print('{}--->{}'.format(img,zhong_result))
            f3.write('{}--->{}\n'.format(img,zhong_result))
            matrix[hang][lie]+=1
        #print(img)
        #print(zhong_result)
    #break
    print("{}种的测试集长度：{}\n".format(shu,zhong_num))
    print('{}种:\nAcc:{:.6f}\n'.format(shu,zhong_acc/zhong_num))
    f.write('{}种{:.6f}\n'.format(shu,zhong_acc/zhong_num))
f2.writelines(str(matrix.tolist()))
print('18属分类准确率为{:.2f}%'.format(shu_r_num/1.5))
print('二级融合网络准确率为{:.2f}%'.format(r_num/1.5))
f.close()
f2.close()
f3.close()