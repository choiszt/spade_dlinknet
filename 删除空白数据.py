import os

imgline=[]
maskline=[]
with open('./baidu/img_paths_txt/total_img.txt', 'r', encoding='utf8') as f:
    linea = f.readlines()  # baidu/河南省/平顶山市/mask/4458.png
for img in linea:
    imgline.append(img.strip())

# ROOT=line
with open('./baidu/img_paths_txt/total_mask.txt', 'r', encoding='utf8') as f:
    lineb = f.readlines()
for img in lineb:
    maskline.append(img.strip())
numberlist=[]
for i in range(2425,2565):
    numberlist.append(str(i))
print(numberlist)
for i in imgline:
    if i[-8:-4] not in numberlist:
        with open('./baidu/img_paths_txt/1.txt', 'a', encoding='utf8') as f:
            f.write(i+'\n')
# imgline.remove()
for i in maskline:
    if i[-8:-4] not in numberlist:
        print(i+'\n')
        with open('./baidu/img_paths_txt/2.txt', 'a', encoding='utf8') as f:
            f.write(i+'\n')