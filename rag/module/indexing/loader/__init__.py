from .doc_loader import CustomizedOcrDocLoader
from .pdf_loader import CustomizedPyMuPDFLoader
from .pptx_loader import CustomizedPPTXLoader
from .web_loader import CustomizedWebBaseLoader

DOCUMENTS_LOADER_MAPPING = {
    "CustomizedPyMuPDFLoader": [".pdf"],
    "UnstructuredFileLoader": [".txt"],
    "CustomizedOcrDocLoader": [".docx"],
    "CustomizedPPTXLoader": [".pptx"],
    "UnstructuredPowerPointLoader": [".ppt"],
    "CustomizedWebBaseLoader": [".html", ".htm"],
}