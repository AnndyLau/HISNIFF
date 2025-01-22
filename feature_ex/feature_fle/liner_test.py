import os
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import time
from wepub import SendMessage
from liner_model import SimpleNet,SimpleNet2
sm = SendMessage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = []
        self.labels = []
        
        # 读取数据文件
        with open(data_file, 'r') as f:
            for line in f:
                line = eval(line.strip())#.split('\n')  # 假设数据以空格分隔
                sequence = [float(x) for x in line]
                #print(len(sequence))
                self.data.append(sequence)
        
        # 读取标签文件
        with open(label_file, 'r') as f:
            for line in f:
                label = int(line.strip())
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data), torch.tensor(label)
    def get_class_to_idx(self):
        # 返回类别到标签的映射关系
        return self.class_to_idx
batch_size = 128
num_class=9
shu = 'Scaptonyx'#['Scapanus','Scaptonyx','Uropsilus','Euroscaptor','Mogera']
lei = 'concat_5:1'
server_name = 'al-5epoch'
name_style = shu+'/'+lei
start_time = time.time()
file_path = './'+server_name+'/'+name_style+'_data/'
#file_path = '/data_16t/zengweiqi/data_al/test_data/18/concat_5:1_data/'
x_test_path = file_path+'x_test.txt'
y_test_path = file_path+'y_test.txt'
test_dataset = CustomDataset(x_test_path, y_test_path)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
test_data_size = len(test_dataset)
print('测试集数量为:{}'.format(test_data_size))
net = torch.load('./weights-'+server_name+"/"+name_style+'/'+'best_model.pkl',map_location=device)
loss_func = torch.nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
eval_loss = 0
eval_acc = 0
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = net(imgs)
        loss = loss_func(outputs, targets)
        eval_loss += loss.item() * targets.size(0)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == targets).sum()
        eval_acc += num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_size
                                                    , eval_acc / test_data_size))