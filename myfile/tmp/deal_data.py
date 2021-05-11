import os

def get_label(img_path):
    pos_2 = img_path.rfind(".")
    pos_1 = img_path.rfind("_")
    label = img_path[pos_1+1:pos_2]
    return label


suffix_list = [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]

def check_file_time(path):
    cnt_ = 0
    imagePathList = []
    for root, dir, files in os.walk(path):
        for file in files:
            try:
                full_path = os.path.join(root, file)


                suffix_img = full_path.rsplit(".",1)
                if(len(suffix_img)<2):
                    continue
                suffix_img = "." + suffix_img[-1]
                if suffix_img in suffix_list:
                    cnt_ += 1
                    print (cnt_, "  :: ", full_path)
                    imagePathList.append(full_path)

            except IOError:
                continue

    return imagePathList


dir_img = "/media/pc/data_1/data/rec_general_data/data/test"

# dir_img = "/data_1/2020biaozhushuju/20210508_generate/data/train"

img_list = check_file_time(dir_img)

with open("./test.txt","w")as fw:
    for cnt,path in enumerate(img_list):
        # print(cnt,path)


        label = get_label(path)

        print(cnt, "   ::", path + "  :: " + label)

        fw.write(path+"::"+label + "\n")

