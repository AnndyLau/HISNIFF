import torch
import torchvision
from torchvision import models,transforms
import os
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from torch import nn
from PIL import Image
import numpy as np
import torch.cuda.amp as amp
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
d_acc = {}
d_count = {}
class_18 = ['Condylura', 'Desmana', 'Dymecodon', 'Euroscaptor', 'Galemys', 'Mogera', 
            'Neurotrichus', 'Orescaptor', 'Parascalops', 'Parascaptor', 'Scalopus', 
            'Scapanulus', 'Scapanus', 'Scaptochirus', 'Scaptonyx', 'Talpa', 'Uropsilus', 'Urotrichus']
class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
            'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 
            'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 'Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
            'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
            'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
            'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
            'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
            'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
# class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris','Euroscaptor grandis', 
#             'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 
#             'Euroscaptor orlovi', 'Euroscaptor parvidens', 'Galemys pyrenaicus', 'Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
#             'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
#             'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
#             'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 
#             'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
#             'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
# class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
#             'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 
#              'Euroscaptor parvidens', 'Galemys pyrenaicus', 'Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
#             'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
#             'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
#             'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
#             'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
#             'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
# class_49 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
#             'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 
#             'Galemys pyrenaicus','Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
#             'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
#             'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
#             'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
#             'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
#             'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
for i in class_18:
    d_acc[i] = 0
    d_count[i] = 0
#l_class_18 = list(class_18.items())
model_list = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
num_list = [5,7,3,4,3,9,9]
zhong_model_list = [['Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes'],
                    ['Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 'Talpa talyschensis'],
                    #['Scaptonyx fusicaudus', 'Scaptonyx sp3'],
                    ['Scaptonyx Scaptonyx sp1', 'Scaptonyx fusicaudus', 'Scaptonyx sp3'],
                    ['Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii'],
                    ['Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    #['Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    #['Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    ['Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura'],
                    #['Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
                    #['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor parvidens']]
                    #['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi']]
                    ['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
input_size=600
transform = torchvision.transforms.Compose(
    [
            transforms.Resize((input_size,input_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
lei = 'teeth'
class_num='1'
#class_path =  "../new_data_al/"+lei+"_al_49_new/test"  #测试集
class_path =  "./"+lei+"_5zhe_data/"+lei+"_data_"+class_num+"/"+lei+"_51_r_p/test"  #测试集
class_path = '/home/cz/data/final_data/teeth_data/teeth_r_p_od'
#server_name = 'b5-al-head'
#class_path =  "./teeth_al_od/teeth_al_49_od/test"  #测试集

class_dir = os.listdir(class_path)
max_allacc = 0
test_num =0
test_acc_list = []
matrix = np.zeros((len(class_49),len(class_49)))
txt_path = './feature_ex/datatxt_b7_'+class_num+'/'
server_name = 'data18-51zhong-'+lei+'_od'
if os.path.exists(txt_path) is False:
    os.makedirs(txt_path)
f= open(txt_path+server_name+'.txt',mode = 'w')
f2 = open(txt_path+server_name+'-matrix.txt', mode = 'w')
f3 = open(txt_path+server_name+'-error.txt',mode='w')
f4 = open(txt_path+server_name+'-y_ture.txt',mode='w') 
f5 = open(txt_path+server_name+'-y_pred.txt',mode='w')
server_name = 'b7-'+lei
shu_18_path = './'+lei+'_5zhe_data/'+lei+'_data_'+class_num+'/weights-'+server_name+'/EfficientNet_50_18/best_network_eb4.pth'#18属分类模型存放位置
shu_acc = 0
acc_count_all =0
for classname in class_49:#class_49:#51种
    num_class = 18 
    data_path = os.path.join(class_path,classname)
    #net = models.efficientnet_b4().to(device)
    
    net = EfficientNet.from_name('efficientnet-b7').to(device)
    #net = EfficientNet.from_pretrained('efficientnet-b4')
    net_class = str(net).split('(')[0]#网络名称
    
    acc_count = 0
    flag = 0
    image_path = os.path.join(data_path)
    image_data = os.listdir(image_path)
    img_num = len(image_data)
    #net.classifier[1] = nn.Linear(1792,num_class)
    
    test_list = []   
    print("{}种的测试集长度：{}".format(classname,img_num))
    for img in image_data:#图片
        num_ftrs = net._fc.in_features
        net._fc = nn.Linear(num_ftrs, num_class)
        #net.classifier[1] = nn.Linear(1792,num_class)
        net.load_state_dict(torch.load(shu_18_path,map_location=device))
        ipath = os.path.join(image_path,img)
        image = Image.open(ipath)
        image = image.convert('RGB')
        image = transform(image)
        image = torch.reshape(image, (1, 3, input_size, input_size))
        image = image.to(device)
        net.eval()
        net.to(device)
        with torch.no_grad():
            output = net.forward(image)
        result = output.argmax(1)
        result1 = str(result).split('[')[1].split(']')[0]
        test_list.append(result1)
        #print(output)
        #print('{}图片为{}属'.format(img,class_18[eval(result1)]))
        shu = class_18[eval(result1)]
        r_shu = classname.split(' ')[0]
        d_count[r_shu]+=1
        f4.write('{}\n'.format(class_49.index(classname)))
        if(r_shu==shu):
            #print('判断正确')
            shu_acc+=1
            pass
        else:
            print('{}---->{}'.format(img,shu))
            f3.write('{}---->{}\n'.format(img,shu))
            matrix[class_49.index(classname)][len(class_49)-1]+=1
            f5.write(str(len(class_49)-1)+'\n')
            continue
        if(shu in model_list):
            shu_num = num_list[model_list.index(shu)]
            num_ftrs = net._fc.in_features
            net._fc = nn.Linear(num_ftrs, shu_num)
            #net.classifier[1] = nn.Linear(1792,shu_num)
            name_style = net_class+"b7-"+lei+"_50_"+str(shu_num)+'_'+shu#命名格式
            net.load_state_dict(torch.load('./'+lei+'_5zhe_data/'+lei+'_data_'+class_num+'/weights-'+server_name+'/'+name_style+'/best_network_eb4.pth',map_location=device))
            net.to(device)
            with torch.no_grad():
                output = net(image)
            result = output.argmax(1)
            result1 = str(result).split('[')[1].split(']')[0]#对应种分类器中的哪一种
            zhong =  zhong_model_list[model_list.index(shu)][eval(result1)]
            r_zhong = classname
            #print('该图片为{}种'.format(zhong))
            hang = class_49.index(r_zhong)
            lie = class_49.index(zhong)
            #print(hang,lie)
            #print(matrix[hang][lie])
            matrix[hang][lie]+=1
            f5.write('{}\n'.format(class_49.index(zhong)))
            if(zhong == r_zhong):
                #print('判断正确')
                acc_count+=1
                d_acc[shu]+=1
                
            else:
                print('{}---->{}'.format(img,zhong))
                f3.write('{}---->{}\n'.format(img,zhong))
        else:
            zhong = classname
            hang = class_49.index(zhong)
            lie = class_49.index(zhong)
            matrix[hang][lie]+=1
            acc_count+=1
            f5.write('{}\n'.format(class_49.index(zhong)))
            #print('该图片为{}种'.format(zhong))
        
    test_num+=img_num
    acc_count_all+=acc_count
    #ave_allacc+= acc_count/img_num*test_data_size/227
    
    print('{}种:\nAcc:{:.6f}\n'.format(classname,acc_count/img_num))
    test_acc_list.append(acc_count/img_num)
    f.write('{}种{}\n'.format(classname,acc_count/img_num))
max_allacc+= acc_count_all/test_num
f2.writelines(str(matrix.tolist()))
print("18属分类准确率为:{:.2f}%\n".format(shu_acc/test_num*100))
print("网络测试总精度为:{:.2f}%\n".format(max_allacc*100))
print("总检测样本量为{}".format(test_num))
for shu in model_list:
    print("{}属检测精度为:{:.6f}".format(shu,d_acc[shu]/d_count[shu]))
f.close()
f2.close()
f3.close()
f4.close()
f5.close()
