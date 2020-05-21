import os

if __name__ == '__main__':
    f = open('data/annotations.txt', 'w+')
    input_path = 'data/VOC2012/JPEGImages/'
    target_path = 'data/VOC2012/SegmentationObject/'
    for file in os.listdir(input_path):
        name = file.split('.')[0]
        if not os.path.isfile(target_path + name + '.png'):
            continue
        f.write(input_path + name + '.jpg' + '\t' + target_path + name + '.png' + '\n')
    f.close()
