import os
import io
from datetime import datetime
from fpdf import FPDF
from PIL import Image


def safe(text):
    return (str(text)
        .replace('\u2014', '-').replace('\u2013', '-')
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2192', '->').replace('\u2022', '-')
    )


def build_pdf(results, disease_info, class_names):
    PAGE_W = 210; PAGE_H = 297; M = 12
    CONT_W = PAGE_W - M * 2; IMG_W = 68
    COL2_X = M + IMG_W + 6; COL2_W = PAGE_W - COL2_X - M
    FOOTER_Y = PAGE_H - 11; MAX_Y = PAGE_H - 20

    pdf = FPDF(unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)

    for rec_idx, r in enumerate(results):
        pdf.add_page()
        info = disease_info[r['disease']]

        # Header
        pdf.set_fill_color(13, 59, 59)
        pdf.rect(0, 0, PAGE_W, 22, 'F')
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(M, 5)
        pdf.cell(CONT_W, 7, safe('Cattle Health Check  -  Disease Report'))
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(156, 163, 175)
        pdf.set_xy(M, 14)
        pdf.cell(CONT_W, 5, safe(
            f'Date: {datetime.now().strftime("%d %B %Y  %I:%M %p")}  |  '
            f'Animal {rec_idx+1} of {len(results)}  |  '
            f'Model: {r.get("model_version", "v1.0")}  |  '
            f'File: {r["filename"]}'
        ))

        y = 26

        # Image
        img_h_mm = 0
        if r.get('image_bytes'):
            img_path = f'/tmp/_cattle_img_{rec_idx}.jpg'
            pil_img  = Image.open(io.BytesIO(r['image_bytes'])).convert('RGB')
            w, h     = pil_img.size
            px       = int(IMG_W * 3.7795)
            if w > px:
                pil_img = pil_img.resize((px, int(h * px / w)), Image.LANCZOS)
            img_h_mm = IMG_W * (pil_img.size[1] / pil_img.size[0])
            pil_img.save(img_path, format='JPEG', quality=55, optimize=True)
            pdf.image(img_path, x=M, y=y, w=IMG_W)
            os.remove(img_path)

        # Right column
        ry = y
        pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(13, 59, 59)
        pdf.set_xy(COL2_X, ry); pdf.multi_cell(COL2_W, 6.5, safe(info['full_name']))
        ry = pdf.get_y()
        pdf.set_font('Helvetica', '', 8); pdf.set_text_color(107, 114, 128)
        pdf.set_xy(COL2_X, ry)
        pdf.cell(COL2_W, 5, safe(f'Confidence: {r["confidence"]:.1f}%'))
        ry += 6
        pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(15, 118, 110)
        pdf.set_xy(COL2_X, ry); pdf.cell(COL2_W, 5, safe(info['urgency_msg']))
        ry += 7
        for lbl, val in [('Vet Required', 'Yes' if info['requires_vet'] else 'No'), ('Status', info['severity'])]:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(COL2_X, ry); pdf.cell(28, 5, safe(lbl + ':'), ln=False)
            pdf.set_font('Helvetica', '', 8); pdf.set_text_color(55, 65, 81)
            pdf.cell(COL2_W - 28, 5, safe(val)); ry += 5.5

        y = max(y + img_h_mm, ry) + 5
        pdf.set_draw_color(209, 213, 219); pdf.line(M, y, PAGE_W - M, y); y += 4

        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'What You Are Seeing:'); y += 5.5
            pdf.set_font('Helvetica', 'I', 8); pdf.set_text_color(75, 85, 99)
            pdf.set_xy(M, y); pdf.multi_cell(CONT_W, 4.5, safe(info['what_you_see'])); y = pdf.get_y() + 3

        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'What To Do:'); y += 5.5
            pdf.set_fill_color(230, 244, 244); pdf.set_font('Helvetica', '', 8); pdf.set_text_color(55, 65, 81)
            pdf.set_xy(M, y); pdf.multi_cell(CONT_W, 4.5, safe(info['what_to_do']), fill=True); y = pdf.get_y() + 5

        LABEL_W = 78; PCT_W = 16
        BAR_X = M + LABEL_W + PCT_W; BAR_W = CONT_W - LABEL_W - PCT_W
        if y < MAX_Y:
            pdf.set_font('Helvetica', 'B', 8); pdf.set_text_color(107, 114, 128)
            pdf.set_xy(M, y); pdf.cell(CONT_W, 5, 'Analysis Breakdown:'); y += 6
            for cname in class_names:
                if y >= MAX_Y: break
                pct    = r['all_predictions'].get(cname, 0)
                is_top = cname == r['disease']
                fill_w = BAR_W * (pct / 100.0)
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5); pdf.set_text_color(31, 41, 55)
                pdf.set_xy(M, y)
                pdf.cell(LABEL_W, 5, safe(('> ' if is_top else '   ') + disease_info[cname]['full_name']), ln=False)
                pdf.set_font('Helvetica', 'B' if is_top else '', 7.5); pdf.set_text_color(15, 118, 110)
                pdf.cell(PCT_W, 5, f'{pct:.1f}%', ln=False)
                bar_y = y + 1.2
                pdf.set_fill_color(229, 231, 235); pdf.rect(BAR_X, bar_y, BAR_W, 2.8, 'F')
                if fill_w > 0.1:
                    pdf.set_fill_color(15, 118, 110) if is_top else pdf.set_fill_color(156, 163, 175)
                    pdf.rect(BAR_X, bar_y, fill_w, 2.8, 'F')
                y += 6.5

        # Footer
        pdf.set_xy(M, FOOTER_Y)
        pdf.set_font('Helvetica', 'I', 6); pdf.set_text_color(140, 140, 140)
        pdf.cell(CONT_W, 4,
            safe('AI-assisted only. Does NOT replace a veterinary diagnosis. '
                 'Always consult a licensed vet.  |  Cattle Health Check Kenya'),
            align='C')

    return bytes(pdf.output())