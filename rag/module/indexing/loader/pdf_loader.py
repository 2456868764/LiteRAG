from abc import ABC
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.base import BaseLoader

import cv2

from comps import CustomLogger
from rag.module.indexing.loader.ocr import get_rapid_ocr
import tqdm
import numpy as np


logger = CustomLogger("pdf_loader")

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.3, 0.3)



class CustomizedPyMuPDFLoader(BaseLoader, ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        def pdf2text(filepath):
            import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            ocr = get_rapid_ocr()
            doc = fitz.open(filepath)
            ocr_results=[]
            # print("total page count:", doc.page_count)
            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page dataprep: 0")
            # print(f"pdf doc len:{len(doc)}")
            for i, page in enumerate(doc):
                # print(f"dataprep page:{i}")
                b_unit.set_description("RapidOCRPDFLoader context page dataprep: {}".format(i))
                b_unit.refresh()
                ocr_results.append("")
                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                                or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            # tmp_img = Image.fromarray(img_array);
                            ori_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            ocr_results[i] += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return ocr_results

        def rotate_img(img, angle):
            '''
            img   --image
            angle --rotation angle
            return--rotated img
            '''

            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img

        # 使用 PyMuPDFLoader 加载 PDF
        ocr_results = pdf2text(self.file_path)
        logger.info(f"total ocr results:{len(ocr_results)}")
        logger.info(ocr_results)
        loader = PyMuPDFLoader(self.file_path)
        data = loader.load()
        logger.info(f"total data page:{len(data)}")
        # 聚合所有页面内容
        aggregated_content = ""
        i = 0
        for page in data:
            aggregated_content += page.page_content + ocr_results[i]
            i += 1
        # 创建一个 Document 对象，包含聚合后的内容
        doc = Document(page_content=aggregated_content, metadata={"source": self.file_path})
        return [doc]  # 返回一个包含文档的列表



# 示例用法
if __name__ == "__main__":
    # 指定 PDF 文件路径
    pdf_file_path = "./data/raw/pdf/example.pdf"

    # 创建自定义加载器实例
    custom_loader = CustomizedPyMuPDFLoader(pdf_file_path)

    # 加载并获取文档
    documents = custom_loader.load()

    # 打印文档内容
    for doc in documents:
        print(doc.page_content)
