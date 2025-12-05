from EnhanceCrack import batch_enhance_binary_cracks
from RandomImage import delete_files_in_folder, moveFile
from StandardizeCrack import batch_process_crack_images

fileDir_Image = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceImage\\"  # Image源图片文件夹路径
fileDir_Mask = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\SourceImageMask\\SourceMask\\"  # Mask源图片文件夹路径
tarDir_Image_Train = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainImage\\'  # Image移动到新的训练文件夹路径
tarDir_Image_Test = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TestImageMask\\TestImage\\'  # Image移动到新的测试文件夹路径
tarDir_Mask_Train = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TrainImageMask\\TrainMask\\'  # Mask移动到新的训练文件夹路径
tarDir_Mask_Test = 'C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\TestImageMask\\TestMask\\'  # Mask移动到新的测试文件夹路径
standardize_output_folder = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\Standardize_processed_results\\" # 标准化输出文件夹
enhancement_output_folder = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\Enhancement_processed_results\\" # 增强输出文件夹
pre_results_folder = "C:\\Users\\dell\\PycharmProjects\\SAM2-UNeXT\\PreResults\\" # test结果文件夹

target_width=5  # 标准化裂缝宽度

"""清空数据集文件夹"""
delete_files_in_folder(tarDir_Image_Train)
delete_files_in_folder(tarDir_Image_Test)
delete_files_in_folder(tarDir_Mask_Train)
delete_files_in_folder(tarDir_Mask_Test)
delete_files_in_folder(standardize_output_folder)
delete_files_in_folder(enhancement_output_folder)
delete_files_in_folder(pre_results_folder)

"""标准化掩膜裂缝宽度"""
batch_process_crack_images(
    fileDir_Mask,
    standardize_output_folder,
    target_width
)

"""掩膜裂缝增强"""
# batch_enhance_binary_cracks(standardize_output_folder, enhancement_output_folder)

"""随机划分数据集"""
# fileDir = [fileDir_Image, fileDir_Mask, tarDir_Image_Train, tarDir_Image_Test, tarDir_Mask_Train, tarDir_Mask_Test]  # 不标准化不增强
fileDir = [fileDir_Image, standardize_output_folder, tarDir_Image_Train, tarDir_Image_Test, tarDir_Mask_Train, tarDir_Mask_Test]  # 仅标准化不增强
# fileDir = [fileDir_Image, enhancement_output_folder, tarDir_Image_Train, tarDir_Image_Test, tarDir_Mask_Train, tarDir_Mask_Test]  # 既标准化又增强
moveFile(fileDir)
