import io
# import os
from pathlib import Path
import numpy as np
import PIL 
from reportlab.platypus import Frame
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageTemplate
from reportlab.platypus import BaseDocTemplate, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, Paragraph, Image
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import matplotlib.pyplot as plt
import cv2

PAGESIZE = A4
PADDING = {"leftPadding": 72, "rightPadding": 72,
           "topPadding": 72, "bottomPadding": 18}
ANGLE_ROUND = 4

PORTRAIT_FRAME = Frame(0, 0, *PAGESIZE, **PADDING)

def on_page(canvas, doc, pagesize=A4):
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(pagesize[0]/2, 25, str(page_num))

PORTRAIT_TEMPL = PageTemplate(
id='portrait',
frames=PORTRAIT_FRAME,
onPage=on_page,
pagesize=PAGESIZE)

def img2pdfimg(img, downscale=2):
    if downscale != 1:
        img = cv2.resize(img, (0, 0), fx=(1 / downscale), fy=(1 / downscale))
    img = PIL.Image.fromarray(img)
    # plt.imshow(img)
    # plt.show()
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_reader = ImageReader(buf)
    img_width, img_height = img_reader.getSize()
    aspect = img_height / float(img_width)
    # print(img_width, img_height, aspect)

    # Calculate the dimensions to fit on the page
    width = PAGESIZE[0] - PADDING["leftPadding"] - PADDING["rightPadding"]
    height = aspect * width
    return Image(buf, width, height)


def create_final_report(report_dir, full_report_name, plots4final_report):

    FREP_NAME = Path(report_dir, f"{full_report_name}").as_posix()

    doc_name = f"{FREP_NAME}.pdf"
    doc = BaseDocTemplate(
        doc_name,
        pageTemplates=[
            PORTRAIT_TEMPL,
        ])
    
    styles = getSampleStyleSheet()
    styles['Heading3'].textColor = colors.gray
    styles['Heading3'].fontName = 'Helvetica'

    story = [
    Paragraph('Binarization Final Report', styles['Heading1']),
    ]

    for i, img in enumerate(plots4final_report):
        img_paragraph = img2pdfimg(img)
        text_paragraphs = []
        text_paragraphs.append(Paragraph(f'Fig. {i + 1}.', styles['Heading3']))        
        story.append(KeepTogether([text_paragraphs[0], img_paragraph]))

    doc.build(story)


    