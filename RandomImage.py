import os, random, shutil, numpy as np


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def moveFile(fileDir):
    Test_rate = 0.2  # 自定义抽取图片的比例
    if len(os.listdir(fileDir[0])) == len(os.listdir(fileDir[1])):
        filenumber = len(os.listdir(fileDir[0]))
        population = np.arange(0, filenumber)
        print(population)
        picknumber = int(filenumber * Test_rate)  # 按照rate比例从文件夹中取一定数量图片
        sample_1 = random.sample(list(population), picknumber)
        print(sample_1)
        sample_2 = list(set(population) - set(sample_1))
        print(sample_2)
        name_1 = os.listdir(fileDir[0])
        name_2 = os.listdir(fileDir[1])
        for i in sample_1:
            shutil.copy(fileDir[0] + name_1[i], fileDir[3] + name_1[i])
            shutil.copy(fileDir[1] + name_2[i], fileDir[5] + name_2[i])
        for i in sample_2:
            shutil.copy(fileDir[0] + name_1[i], fileDir[2] + name_1[i])
            shutil.copy(fileDir[1] + name_2[i], fileDir[4] + name_2[i])
    else:
        print("图片数量不匹配")
    return


if __name__ == '__main__':
    fileDir_Image = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceImage\\"  # Image源图片文件夹路径
    fileDir_Mask = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceMask\\"  # Mask源图片文件夹路径
    tarDir_Image_Train = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainImage\\'  # Image移动到新的训练文件夹路径
    tarDir_Image_Test = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TestImageMask\\TestImage\\'  # Image移动到新的测试文件夹路径
    tarDir_Mask_Train = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainMask\\'  # Mask移动到新的训练文件夹路径
    tarDir_Mask_Test = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TestImageMask\\TestMask\\'  # Mask移动到新的测试文件夹路径
    fileDir = [fileDir_Image, fileDir_Mask, tarDir_Image_Train, tarDir_Image_Test, tarDir_Mask_Train, tarDir_Mask_Test]
    delete_files_in_folder(tarDir_Image_Train)
    delete_files_in_folder(tarDir_Image_Test)
    delete_files_in_folder(tarDir_Mask_Train)
    delete_files_in_folder(tarDir_Mask_Test)
    moveFile(fileDir)
















