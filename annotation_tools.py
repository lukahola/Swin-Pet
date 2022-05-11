import  os

def write_name(file_path, txt_path):
    file_list = os.listdir(file_path)
    num = len(file_list)
    i = 1
    with open(txt_path, 'w') as f:
        for file_name in file_list:
            image_list = os.listdir(file_path + '/' + file_name)
            for image_name in image_list:
                f.write(file_name + '/' + image_name + ' ' + str(i) + '\n')
            i += 1




if __name__ == '__main__':

    file_path = 'G:/ln/imagenet/val'
    txt_path = 'G:/ln/imagenet/val_map.txt'
    write_name(file_path, txt_path)