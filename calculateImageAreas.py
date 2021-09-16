import os
import cv2
import xlsxwriter

"""
计算各类地表面积
1 building: 0 128 128
2 forest: 0 128 0
3 grass: 0 0 128
4 water: 128 0 0
5 wetland: 128 0 128
6 lake： 128 128 0
"""
dir_path = 'data/after_process_label/predictResultLabel/'
calculate_result = {'grass': 0, 'forest': 0, 'wetland': 0, 'river': 0, 'lake': 0, 'building': 0, 'other': 0}


def writer_into_excel():
    xls_name = '地表分类面积预测结果.xlsx'  # 定义xlsx文件名称
    workbook = xlsxwriter.Workbook(xls_name)  # 创建xlsx文件，若已存在则覆盖。
    worksheet = workbook.add_worksheet('面积统计结果')
    row = 0
    col = 0

    # 按照行和列写入数据
    for item, cost in calculate_result.items():
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, cost)
        row += 1

    # 关闭并保存表格内容
    workbook.close()


def getOneImageArea(imagePath):
    grassArea, forestArea, wetlandArea, riverArea, lakeArea, buildingArea, other = 0, 0, 0, 0, 0, 0, 0
    img = cv2.imread(imagePath)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 1:
                buildingArea = buildingArea + 1
            elif img[i, j, 0] == 2:
                forestArea = forestArea + 1
            elif img[i, j, 0] == 3:
                grassArea = grassArea + 1
            elif img[i, j, 0] == 4:
                riverArea = riverArea + 1
            elif img[i, j, 0] == 5:
                wetlandArea = wetlandArea + 1
            elif img[i, j, 0] == 6:
                lakeArea = lakeArea + 1
            else:
                other = other + 1
    return grassArea, forestArea, wetlandArea, riverArea, lakeArea, buildingArea, other


def calculateArea(image_list):
    grassArea, forestArea, wetlandArea, riverArea, lakeArea, buildingArea, otherArea = 0, 0, 0, 0, 0, 0, 0
    for image_name in image_list:
        image_path = dir_path + image_name
        grass, forest, wetland, river, lake, building, other = getOneImageArea(image_path)
        grassArea = grassArea + grass
        forestArea = forestArea + forest
        wetlandArea = wetlandArea + wetland
        riverArea = riverArea + river
        lakeArea = lakeArea + lake
        buildingArea = buildingArea + building
        otherArea = otherArea + other
    calculate_result['grass'] = grassArea
    calculate_result['forest'] = forestArea
    calculate_result['wetland'] = wetlandArea
    calculate_result['river'] = riverArea
    calculate_result['lake'] = lakeArea
    calculate_result['building'] = buildingArea
    calculate_result['other'] = otherArea
    print(calculate_result)
    writer_into_excel()


def get_Image_List(dirPath):
    image_name_list = []
    for root, dirs, files in os.walk(dirPath):
        image_name_list = files
    image_name_list.sort(key=lambda x: int(x[:-4]))
    calculateArea(image_name_list)


get_Image_List(dir_path)


