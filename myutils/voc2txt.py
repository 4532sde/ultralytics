import xml.etree.ElementTree as ET  # xml解析包
import os

sets = ['train', 'test', 'val']
classes = ['face','face_mask']

# 进行归一化操作
def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return (x, y, w, h)  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


# year ='2012', 对应图片的id（文件名）
def convert_annotation(root, image_id):
    '''
    将对应文件名的xml文件转化为label文件，xml文件包含了对应的bunding框以及图片长宽大小等信息，
    通过对其解析，然后进行归一化最终读到label文件中去，也就是说
    一张图片文件对应一个xml文件，然后通过解析和归一化，能够将对应的信息保存到唯一一个label文件中去
    labal文件中的格式: calss x y w h，同时，一张图片对应的类别有多个，所以对应的buinding的信息也有多个
    '''
    # 对应的通过year 找到相应的文件夹，并且打开相应image_id的xml文件，其对应bund文件
    in_file = open(os.path.join(root, 'Annotations', '%s.xml' % image_id), encoding='utf-8')
    # 准备在对应的image_id 中写入对应的label，分别为
    # <object-class> <x> <y> <width> <height>
    out_file = open(os.path.join(root, 'labels', '%s.txt' % image_id), 'w', encoding='utf-8')
    # 解析xml文件
    tree = ET.parse(in_file)
    # 获得对应的键值对
    root = tree.getroot()
    # 获得图片的尺寸大小
    size = root.find('size')
    # 如果xml内的标记为空，增加判断条件
    if size is not None:
        # 获得宽
        w = int(size.find('width').text)
        # 获得高
        h = int(size.find('height').text)
        # 遍历目标obj
        for obj in root.iter('object'):
            if w == 0 or h == 0:
            # 打印警告信息，方便您定位问题文件
                print(f"警告: 文件 {image_id}.xml 中的图像尺寸为零 (width: {w}, height: {h})。已跳过此文件。")
            # 直接返回，不再处理这个有问题的xml文件
                return
            # 获得difficult
            if obj.find('difficult') is not None:
                difficult = int(obj.find('difficult').text)
            else:
                difficult = 0
            # 获得类别 =string 类型
            cls = obj.find('name').text
            # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
            if cls not in classes or int(difficult) == 1:
               continue
            # 通过类别名称找到id
            cls_id = classes.index(cls)
            # 找到bndbox 对象
            xmlbox = obj.find('bndbox')
            # 获取对应的bndbox的数组 = ['xmin','xmax','ymin','ymax']
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)
            # 带入进行归一化操作
            # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
            bb = convert((w, h), b)
            # bb 对应的是归一化后的(x,y,w,h)
            # 生成 calss x y w h 在label文件中
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    ROOT = 'E:\datasets'
    for image_set in sets:
        '''
        对所有的文件数据集进行遍历
        做了两个工作:
        1．将所有图片文件都遍历一遍，并且将其所有的全路径都写在对应的txt文件中去 方便定位
        2．同时对所有的图片文件进行解析和转化，将其对应的 bundingbox 以及类别的信息全部解析写到 label 文件中去
        最后再通过直接读取文件 就能找到对应的 label 信息
        '''
        # 先找labels文件夹如果不存在则创建
        if not os.path.exists(os.path.join(ROOT, 'labels')):
            os.makedirs(os.path.join(ROOT, 'labels'))
        # 读取在 ImageSets 中的train、test..等文件的内容
        image_ids = open(os.path.join(ROOT, 'ImageSets', '%s.txt' % image_set)).read().strip().split()
        # 打开对应的.txt 文件对其进行写入准备
        list_file = open(os.path.join(ROOT, '%s.txt' % image_set), 'w')
        # 将对应的文件_id以及全
                # 路径写进去并换行
        for image_id in image_ids:
            list_file.write(os.path.join(ROOT, 'images', '%s.jpg\n' % image_id))
            # 开始解析xml文件的标注格式
            convert_annotation(root=ROOT, image_id=image_id)
        # 关闭文件
        list_file.close()

