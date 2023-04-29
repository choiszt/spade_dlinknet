import shutil
dict={}
gt_dict={}
new={}
cnt=0
sh=0
source='./baidu/groundtruth/'
testtarget='./baidu/gt_dataset/test/'
traintarget='./baidu/gt_dataset/train/'
with open('./rankIOU.txt')as f:
    for i in f.readlines():
        dict[i[:20].rstrip()]=i.rstrip()[20:]
with open('./gt_rankIOU.txt')as f:
    for i in f.readlines():
        gt_dict[i[:20].rstrip()]=i.rstrip()[20:]
for i in gt_dict.keys():
    sh+=1
    if float(dict[i])>=0.5 and float(gt_dict[i])>=0.5:
        shutil.copy(source+i,testtarget+i)
        shutil.copy(source + i[:-3]+'jpg', testtarget + i[:-3]+'jpg')
        print(f'IOU of {i} is changed from {dict[i]} to {gt_dict[i]} ')
        cnt+=1
    else:
        shutil.copy(source+i,traintarget+i)
        shutil.copy(source + i[:-3]+'jpg', traintarget + i[:-3]+'jpg')
print(f'{cnt}/{sh}')