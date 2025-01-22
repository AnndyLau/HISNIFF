import os
import shutil
lei = 'head'
kind_path = os.path.join('./'+lei+'_data/'+lei+'_18/train')
kind_file = os.listdir(kind_path)
to_dir = os.path.join('./'+lei+'_data/'+lei+'_18_r_p/train')
for d in kind_file:  # 一级目录Condylura等
    img_path = os.path.join(kind_path,d)
    to_dir_new = os.path.join(to_dir,d)
    imgs = os.listdir(img_path)
    for i in imgs:  # i: Euroscaptor#grandis#57110#grandis_57110_DSC_3659#s#v.JPG
        new_name = 'nr_'+i
        #os.remove(os.path.join(to_dir,new_name))
        shutil.copy(os.path.join(img_path,i),os.path.join(to_dir_new,new_name))