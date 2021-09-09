import cv2
import xlsxwriter as xw
import matplotlib.pyplot as plt

image_area_size = 30 * 30  # km * km
all_point_number = 7200 * 6800
file_path = '../data/newdataSource/predictResult/'
TEST_SET = ['1.tif', '2.tif', '3.tif', '4.tif', '5.tif', '6.tif', '7.tif', '8.tif', '9.tif', '10.tif']
areas = ['工业用地', '城市住宅用地', '农村住宅用地', '交通用地', '稻田', '灌溉地', '旱地', '园地',
         '乔木林地', '灌木林地', '天然草地', '人工草地', '河', '湖', '池塘']
# 各类别面积
areas_dict = {'工业用地': 0, '城市住宅用地': 0, '农村住宅用地': 0, '交通用地': 0, '稻田': 0, '灌溉地': 0, '旱地': 0, '园地': 0,
              '乔木林地': 0, '灌木林地': 0, '天然草地': 0, '人工草地': 0, '河': 0, '湖': 0, '池塘': 0}
# 各类别单位面积碳密度：kg/m2
carbon_fixation_dict = {'工业用地': 0, '城市住宅用地': 0, '农村住宅用地': 0, '交通用地': 0, '稻田': 15.21, '灌溉地': 27.456, '旱地': 12.115, '园地': 19.06,
                        '乔木林地': 15.76, '灌木林地': 11.12, '天然草地': 7.98, '人工草地': 4.12, '河': 9.1, '湖': 23.42, '池塘': 3.9}
index_to_number = [0 for i in range(0, 16)]


def calculate_areas(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] != 0:
                index_to_number[image[i][j]] = index_to_number[image[i][j]] + 1


path = TEST_SET[0]
img = cv2.imread(file_path + path, cv2.IMREAD_GRAYSCALE)
calculate_areas(img)
# for n in range(len(TEST_SET)):
#     path = TEST_SET[n]
#     img = cv2.imread(file_path + path, cv2.IMREAD_GRAYSCALE)
#     calculate_areas(img)

for k in range(1, len(index_to_number)):
    # areas_dict[areas[k - 1]] = index_to_number[k] / all_point_number * image_area_size  # '{:.3f}'.format(1.23456)
    areas_dict[areas[k - 1]] = '{:.4f}'.format(image_area_size / all_point_number * index_to_number[k])


def calculate_carbon_fixation(item_category, areas_size):
    return '{:.4f}'.format(float(areas_size) * float(carbon_fixation_dict[item_category]) * 1e6)


def data_write(data, fileName):
    workbook = xw.Workbook(filename=fileName)
    worksheet1 = workbook.add_worksheet("地表类别统计")
    worksheet1.activate()
    title = ['序号', '类别', '面积(单位km²）', '碳储量（单位kg）']
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    k = 1
    for name, number in data.items():
        fixation_number = calculate_carbon_fixation(name, number)
        insertData = [k, name, number, fixation_number]
        row = 'A' + str(k + 1)
        worksheet1.write_row(row, insertData)
        k += 1
    workbook.close()


fileName = '地表面积和碳储量统计.xlsx'
data_write(areas_dict, fileName)

# 画图

# area_list = [areas for item, areas in areas_dict.items()]
# number_list = [number for item, number in carbon_fixation_dict.items()]
#
# x = list(range(len(area_list)))
#
# total_width, n = 0.4, 1
# width = total_width / n
# plt.bar(x, area_list, width=width, label='面积', tick_label=areas, fc='y')
#
# plt.legend()
# plt.show()
# plt.savefig('area_plot1')
#
#
# total_width, n = 0.4, 1
# width = total_width / n
# plt.bar(x, number_list, width=width, label='碳储量', tick_label=areas, fc='y')
# plt.legend()
# plt.show()
# plt.savefig('carbon_plot1')
