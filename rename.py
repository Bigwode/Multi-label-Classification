# coding:utf-8
import os


# 修改标签格式为需要的格式
names = ['egr', 'man', 'owl', 'puf', 'tou', 'wod']
labels = []
imagePath = '/home/david/caiji/mul/birds/egret/egr001.jpg'
name = imagePath.split(os.path.sep)[-1][:3]
label=int(names.index(name))
labels.append(label)
print(labels)


# 文件夹下文件按顺序重命名函数

# def rename():
#     count=1
#     path="/home/david/caiji/mul/birds/val/wood_duck"
#     filelist=os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
#     for files in filelist:  # 遍历所有文件
#         Olddir=os.path.join(path,files)  # 原来的文件路径
#         if os.path.isdir(Olddir):  # 如果是文件夹则跳过
#             continue
#         filename=os.path.splitext(files)[0]  # 文件名
#         filetype=os.path.splitext(files)[1]  # 文件扩展名
#         Newdir=os.path.join(path,'wod'+str(count+100)+filetype)  # 新的文件路径
#         os.rename(Olddir,Newdir)  # 重命名
#         count+=1
# rename()
