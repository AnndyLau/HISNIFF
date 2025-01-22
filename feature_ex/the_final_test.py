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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
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
# class_49_4 = ['Condylura cristata', 'Desmana moschata', 'Dymecodon pilirostris', 'Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 
#             'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 
#             'Galemys pyrenaicus','Mogera etigo', 'Mogera hainana', 'Mogera imaizumii', 
#             'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta', 'Mogera tokudae', 'Mogera wogura', 'Neurotrichus gibbsii', 
#             'Orescaptor mizura', 'Parascalops breveri', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura', 'Scalopus aquaticus', 
#             'Scapanulus oweni', 'Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii', 'Scaptochirus moschatus', 'Scaptonyx Scaptonyx sp1', 
#             'Scaptonyx fusicaudus', 'Scaptonyx sp3', 'Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 
#             'Talpa talyschensis', 'Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes', 'Urotrichus talpoides']
#l_class_18 = list(class_18.items())
model_list = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
num_list = [5,7,3,4,3,9,9]
shu_tool = []
print('正在读取归一化工具')
class_num = '1'
for shu in model_list:
    teeth_tool = joblib.load('./scaler_dir_b7'+class_num+'/'+shu+'_teeth.pkl')
    head_tool = joblib.load('./scaler_dir_b7'+class_num+'/'+shu+'_head.pkl')
    shu_tool.append([teeth_tool,head_tool])
teeth_tool = joblib.load('./scaler_dir_b7'+class_num+'/18_teeth.pkl')
head_tool = joblib.load('./scaler_dir_b7'+class_num+'/18_head.pkl')
shu_tool.append([teeth_tool,head_tool])
zhong_model_list = [['Uropsilus andersoni', 'Uropsilus gracilis', 'Uropsilus investigator', 'Uropsilus nivatus', 'Uropsilus soricipes'],
                    ['Talpa altaica', 'Talpa aquitania', 'Talpa caeca', 'Talpa europaea', 'Talpa levantis', 'Talpa romana', 'Talpa talyschensis'],
                    #['Scaptonyx fusicaudus', 'Scaptonyx sp3'],
                    ['Scaptonyx Scaptonyx sp1', 'Scaptonyx fusicaudus', 'Scaptonyx sp3'],
                    ['Scapanus latimanus', 'Scapanus occultus', 'Scapanus orarius', 'Scapanus townsendii'],
                    ['Parascaptor Parascaptor sp1', 'Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    #['Parascaptor Parascaptor sp2', 'Parascaptor leucura'],
                    ['Mogera etigo', 'Mogera hainana','Mogera imaizumii', 'Mogera insularis', 'Mogera kanoana', 'Mogera latochei', 'Mogera robusta','Mogera tokudae', 'Mogera wogura'],
                    #['Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
                    #['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor parvidens']]
                    #['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi']]
                    ['Euroscaptor Euroscaptor sp1', 'Euroscaptor grandis', 'Euroscaptor klossi', 'Euroscaptor kuznetsovi', 'Euroscaptor longirostris', 'Euroscaptor malayana', 'Euroscaptor micrura', 'Euroscaptor orlovi', 'Euroscaptor parvidens']]
input_size=600
transform = torchvision.transforms.Compose(
    [
            transforms.Resize((input_size,input_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
head_path = '/home/cz/data/final_data/head_5zhe_data/head_data_'+class_num.split('_')[0]+'/head_51_r_p/test'
#teeth_path = './teeth_al_49_1/test'
teeth_path = '/home/cz/data/final_data/teeth_5zhe_data/teeth_data_'+class_num.split('_')[0]+'/teeth_51_r_p/test'
teeth_path = '/home/cz/data/final_data/teeth_data/teeth_r_p_od'
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
net_18 = torch.load('./feature_file/weights-b7_'+class_num+'/18/concat_1:1/best_model.pkl',map_location=device)
net_18.to(device)
net_18.eval()
r_num = 0
shu_r_num =0
img_num = 0
txt_path = './datatxt_b7_'+class_num+'/'
server_name = 'data18-51-combine_od'
matrix = np.zeros((len(class_49),len(class_49)))
if os.path.exists(txt_path) is False:
    os.makedirs(txt_path)
f= open(txt_path+server_name+'.txt',mode = 'w')
f2 = open(txt_path+server_name+'-matrix.txt', mode = 'w')
f3 = open(txt_path+server_name+'-error.txt',mode='w')
f4 = open(txt_path+server_name+'-y_ture.txt',mode='w') 
f5 = open(txt_path+server_name+'-y_pred.txt',mode='w') 
print(shu_tool)
test_num = 0
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
        head_feature_shu = out_feature_18(teeth_image,head_image,'head',class_num.split('_')[0])
        teeth_feature_shu = out_feature_18(teeth_image,head_image,'teeth',class_num.split('_')[0])
        with torch.no_grad():
            head_feature_shu = process(head_feature_shu)
            teeth_feature_shu = process(teeth_feature_shu)
            concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[-1][0],shu_tool[-1][1])
            shu_output = net_18(concat_feature_shu)
            shu_result = shu_output.argmax(1)
            shu_result = str(shu_result).split('[')[1].split(']')[0]
            shu_result = class_18[int(shu_result)]
            r_shu = shu.split()[0]
            f4.write('{}\n'.format(class_49.index(shu)))
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
                    f5.write('{}\n'.format(class_49.index(shu)))
                    #break
                    continue
            else:
                print('{}---->{}'.format(img,shu_result))
                f3.write('{}---->{}\n'.format(img,shu_result))
                matrix[class_49.index(shu)][len(class_49)-1]+=1
                f5.write(str(len(class_49)-1)+'\n')
                #break
                continue
        shu_id = model_list.index(shu_result)
        num_class = num_list[shu_id]
        head_feature_shu = out_feature(teeth_image,head_image,num_class,'head',shu_result,class_num.split('_')[0])
        teeth_feature_shu = out_feature(teeth_image,head_image,num_class,'teeth',shu_result,class_num.split('_')[0])
        head_feature_shu = process(head_feature_shu)
        teeth_feature_shu = process(teeth_feature_shu)
        concat_feature_shu = guiyihua(teeth_feature_shu,head_feature_shu,shu_tool[shu_id][0],shu_tool[shu_id][1])
        net_49 = torch.load(os.path.join('./feature_file/weights-b7_'+class_num,shu_result,'concat_1:1','best_model.pkl'),map_location=device)
        net_49.eval()
        zhong_output = net_49(concat_feature_shu)
        zhong_result = zhong_output.argmax(1)
        zhong_result = str(zhong_result).split('[')[1].split(']')[0]
        zhong_result = zhong_model_list[shu_id][int(zhong_result)]
        r_zhong = shu
        hang = class_49.index(r_zhong)
        lie = class_49.index(zhong_result)
        if(zhong_result==r_zhong):
            #print('种判断正确')
            r_num+=1
            zhong_acc+=1
            matrix[hang][lie]+=1
            f5.write('{}\n'.format(class_49.index(zhong_result)))
            continue
        else:
            print('{}--->{}'.format(img,zhong_result))
            f3.write('{}--->{}\n'.format(img,zhong_result))
            f5.write('{}\n'.format(class_49.index(zhong_result)))
            matrix[hang][lie]+=1
        #print(img)
        #print(zhong_result)
    #break
    print("{}种的测试集长度：{}\n".format(shu,zhong_num))
    print('{}种:\nAcc:{:.6f}\n'.format(shu,zhong_acc/zhong_num))
    f.write('{}种{:.6f}\n'.format(shu,zhong_acc/zhong_num))
    test_num+=zhong_num
f2.writelines(str(matrix.tolist()))
print('18属分类准确率为{:.2f}%'.format(shu_r_num/test_num*100))
print('二级融合网络准确率为{:.2f}%'.format(r_num/test_num*100))
f.close()
f2.close()
f3.close()
f4.close()
f5.close()