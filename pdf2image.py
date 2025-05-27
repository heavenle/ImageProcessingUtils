import fitz  # PyMuPDF
import glob
import pathlib
import torch
from PIL import Image
import torchvision.transforms as transforms


def pdf_to_images(pdf_path, output_folder, scale_factor=2):
    pdf_path = pathlib.Path(pdf_path)
    filename = pdf_path.stem
    print(f"currently image is {filename}")
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom_matrix = fitz.Matrix(scale_factor, scale_factor)
        pix = page.get_pixmap(matrix=zoom_matrix)
        output_path = f"{output_folder}/{filename}_page_{page_num + 1}.png"
        pix.save(output_path)


transform = transforms.Compose([
    transforms.ToTensor()  # 将 PIL 图像转换为 Tensor
])


def image2pdf(image_files_list, output_pdf):
    # 创建一个新的 PDF 文档
    pdf_document = fitz.open()

    for image_file in image_files_list:
        # 打开图片并获取尺寸
        img = fitz.open(image_file)  # 打开图片文件
        img_width, img_height = img[0].rect.width, img[0].rect.height  # 获取图片尺寸

        # 创建一个新页面，大小与图片一致
        pdf_page = pdf_document.new_page(width=img_width, height=img_height)

        # 将图片插入到页面中
        pdf_page.insert_image(pdf_page.rect, filename=image_file)
    # 保存 PDF 文件
    pdf_document.save(output_pdf)
    pdf_document.close()
    print(f"PDF 文件已生成: {output_pdf}")


if __name__ == "__main__":
    # pdf_list = glob.glob("/home/liyi/DATA/pdf2/pdf/*.pdf")
    pdf_list = [r"D:/文档/期刊/反应条件推荐/Generic_Interpretable_Reaction_Condition.pdf"]
    for pi in pdf_list:
        pdf_to_images(pi, r"D:/文档/期刊/反应条件推荐/Generic_Interpretable_Reaction_Condition", scale_factor=2)

    # image_list = [r"D:/Project/DATA/Chem_retrosynthesis/solve/有机溶剂性质手册_page_3.png"]
    # image2pdf(image_list, r"D:/Project/DATA/Chem_retrosynthesis/solve/2.pdf")
