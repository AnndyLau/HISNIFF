import torch
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
def scaler_tool(shu):
    feature_file = './feature_file/al-5epoch'
    #concat_lei = 'concat_5:1_data'
    teeth_path = feature_file+'/'+shu+'/teeth_data'
    head_path = feature_file+'/'+shu+'/head_data'
    for pathnum,file_path in enumerate([teeth_path,head_path]):
        f=open(os.path.join(file_path,'x_train.txt'),mode='r')
        x_train_list=f.readlines()
        f.close()
        for i in range(len(x_train_list)):
            x_train_list[i]=eval(x_train_list[i].strip())
        if(pathnum==0):
            #print('teeth')
            scaler_teeth = MinMaxScaler(feature_range=(0,1))#牙齿占比
            scaler_teeth.fit(x_train_list)
        else:
            #print('head')
            scaler_head = MinMaxScaler(feature_range=(0,0.2))#头骨占比
            scaler_head.fit(x_train_list)
    return scaler_teeth,scaler_head
def scale_minmax(col,max,min):
    col = np.array(col)
    max = np.array(max)
    min = np.array(min)
    return (col-min)/(max-min).tolist()
# shu_max_min={}
# f = open('scaler.txt',mode='r')
# s = f.readlines()
# shu = '18'
# teeth_max = eval(s[1].split(':')[-1].strip())
# teeth_min =eval(s[2].split(':')[-1].strip())
# head_max =eval(s[3].split(':')[-1].strip())
# head_min =eval(s[4].split(':')[-1].strip())
# ls = [teeth_max,teeth_min,head_max,head_min]
# print(len(ls))
# f = open('scalerall.txt',mode='w',encoding='utf-8')
# model_list = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor','18']
# if os.path.exists(os.path.join('./scaler_dir')) is False:
#     os.makedirs(os.path.join('./scaler_dir'))
# for shu in model_list:
#     scaler_teeth,scaler_head = scaler_tool(shu)
#     joblib.dump(scaler_teeth, os.path.join('./scaler_dir/'+shu+'_teeth.pkl'))
#     joblib.dump(scaler_head, os.path.join('./scaler_dir/'+shu+'_head.pkl'))
#     #f.write('{}\nteeth max:{}\nmin:{}\nhead max:{}\nmin:{}\n'.format(shu,scaler_teeth.data_max_.tolist(),scaler_teeth.data_min_.tolist(),scaler_head.data_max_.tolist(),scaler_head.data_min_.tolist()))

def scaler_values():
    model_list = ['Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
    f = open('scaler.txt',mode='r')
    shu_minmax={}
    s = f.readlines()
    teeth_transform = eval(s[1].split(':')[-1].strip())
    head_transform =eval(s[2].split(':')[-1].strip())
    shu_minmax['18'] = [teeth_transform,head_transform]
    for i in range(1,len(model_list)+1):
        teeth_transform = eval(s[i*3+1].split(':')[-1].strip())
        head_transform =eval(s[i*3+1].split(':')[-1].strip())  
        shu_minmax[model_list[i-1]] = [teeth_transform,head_transform]
    return shu_minmax
def scale_minmax2(col,max,min):
    col = np.array(col)
    max = np.array(max)
    min = np.array(min)
    col*= max
    col+= min
    return col.tolist()
# model_list = ['18','Uropsilus','Talpa','Scaptonyx','Scapanus','Parascaptor','Mogera','Euroscaptor']
# f = open('scaler.txt',mode='w',encoding='utf-8')
# for shu in model_list:
#     tensor = torch.rand(1,2048)
#     teeth_tool,head_tool = scaler_tool(shu)
#     tensor_teeth = teeth_tool.transform(tensor)
#     tensor_head = head_tool.transform(tensor)
#     teeth_transform = tensor/tensor_teeth
#     head_transform = tensor/tensor_head
#     f.write('{}\nteeth_transform:{}\nhead_transform:{}\n'.format(shu,teeth_transform.tolist(),head_transform.tolist()))
# f.close()
# tensor = torch.rand(1,2048)
# teeth_tool,head_tool = scaler_tool('Mogera')
# tensor_teeth = teeth_tool.transform(tensor)
# tensor_head = head_tool.transform(tensor)
# teeth_transform = tensor/tensor_teeth
# head_transform = tensor/tensor_head