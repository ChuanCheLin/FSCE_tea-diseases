#Author: Eric Lin
import json
import os
import shutil
from tqdm import tqdm
# name2id
name2id = { 1:'brownblight', 2:'algal', 3: 'blister', 4: 'sunburn', 5: 'fungi_early', 6: 'roller',
            7: 'moth', 8: 'tortrix', 9: 'flushworm', 10: 'caloptilia', 11: 'mosquito_early', 12: 'mosquito_late',
            13: 'miner', 14: 'thrips', 15: 'tetrany', 16: 'formosa', 17: 'other'}

class json_handler():
    def __init__(self, jpg_data_root, coco_data_root, subset):
        self.full_jpg_dir = jpg_data_root #JPEGImages
        self.data_dir = coco_data_root + '/annotations/instances_' + subset + '.json' #json file
        self.txt_dir = coco_data_root + '/' + subset + '.txt' #txt file
        self.new_jpg_dir = coco_data_root + '/' + subset #train val trainval......
        self.coco_data_root = coco_data_root
        self.subset = subset


    #  get jpg list from arbitrary json file & save as a txt file
    def write_jpg_txt(self):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        images = all_info['images']

        # write txt file
        file = open(self.txt_dir, 'w')       

        # get value
        for i in range(len(images)):
            file.write(images[i].get('file_name'))
            file.write('\n')

        file.close()
    
    # copy jpg file into corresponding coco dataset dir
    def get_jpg_from_txt(self, single_class = False):

        if single_class:
            if(os.path.isdir(self.single_class_jpg_dir)==False):
                os.mkdir(self.single_class_jpg_dir)
            f = open(self.single_class_txt_dir, 'r')
            for jpg in tqdm(f.readlines()):
                jpg = jpg.rstrip('\n')
                ori_path = self.full_jpg_dir + jpg
                new_path = self.single_class_jpg_dir +'/' + jpg
                shutil.copyfile(ori_path, new_path)

            print('jpg files saved at ' + str(self.single_class_jpg_dir))
        else:
            if(os.path.isdir(self.new_jpg_dir)==False):
                os.mkdir(self.new_jpg_dir)

            f = open(self.txt_dir, 'r')
            for jpg in tqdm(f.readlines()):
                jpg = jpg.rstrip('\n')
                ori_path = self.full_jpg_dir + jpg
                new_path = self.new_jpg_dir +'/' + jpg
                shutil.copyfile(ori_path, new_path)

    #  get jpg list from a specific class
    def write_single_class_txt(self, target_class_id = int):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        images = all_info['images']
        annotations = all_info['annotations']
        image_list = []

        # get img list that contains specific class from annotations
        for i in range(len(annotations)):
            id = annotations[i].get('category_id')
            image_id = annotations[i].get('image_id')
            if(id == target_class_id and image_id not in image_list):  
                image_list.append(image_id)

        # write txt file
        dir = self.coco_data_root + '/' + self.subset + '_' + name2id.get(target_class_id) + '.txt'
        self.single_class_jpg_dir = self.coco_data_root + '/' + self.subset + '_' + name2id.get(target_class_id)
        self.single_class_txt_dir = dir
        print('txt file saved at ' + str(dir))
        file = open(dir, 'w') 

        for i in range(len(images)):
            if(images[i].get('id') in image_list):
                file.write(images[i].get('file_name'))
                file.write('\n')
   
if __name__ == "__main__":  
    a = json_handler(
        jpg_data_root= "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/",
        coco_data_root = "/home/eric/mmdetection/data/VOCdevkit/datasets/set1/split2/base/", subset = 'trainvaltest')


    a.write_jpg_txt()
    a.get_jpg_from_txt()

    # a.write_single_class_txt(9)
    # a.get_jpg_from_txt(single_class=True)


