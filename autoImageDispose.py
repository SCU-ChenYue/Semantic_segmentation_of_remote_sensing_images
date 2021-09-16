from predictArea import predict_all_files
from calculateImageAreas import get_Image_List

source_dict_path = 'data/after_process_label/test/'  # 待识别原图所在文件夹
out_path = 'data/after_process_label/predictResult/'  # 预测结果所在文件夹

predict_all_files(source_dict_path, out_path)
get_Image_List(out_path)

