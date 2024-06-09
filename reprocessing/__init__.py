from reprocessing.pdf import PDFDocumentReProcessing
from reprocessing.docx import DocxDocumentReProcessing

pdf_reprocessing = PDFDocumentReProcessing(chunk_size=1000, chunk_overlap=200)
docx_reprocessing = DocxDocumentReProcessing(chunk_size=1000, chunk_overlap=200)

