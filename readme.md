# 该项目是常用的图像处理代码汇总



## 代码汇总：
- common文件
    - remote_img_process_class.py: 遥感图像处理代码，包含图像的读写，转化geojson、shp和tif以及均值方差的计算等。
- covert文件
    - segmentation_format_covert.py: 格式转化代码，实现pie转voc、voc转pie、pie转coco、coco转pie。
- img_16bit_to_8bit.py: 16位图像转8位图像。
- run_img_16bit_to_8bit.sh: linux下自动处理所有tif图像。
- slide_crop.py: 自动裁图代码。
- SegmentationComputerIndex.py: 计算语义分割中的参考指标, [iou, miou, acc]。

