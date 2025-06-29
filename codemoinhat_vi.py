import fitz  # PyMuPDF
import pdfplumber
# import mysql.connector
import re
import os
import time
from datetime import datetime
from tkinter import Tk, filedialog
from PIL import Image, ImageFile
import io
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def connect_to_mysql():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='bb2110065'
    )

def extract_doi(text):
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.IGNORECASE)
    return doi_match.group(0) if doi_match else None

def extract_title_by_fontsize(page):
    max_fontsize = 0
    title_lines = []
    for obj in page.extract_words(use_text_flow=True, extra_attrs=["size"]):
        if obj["size"] > max_fontsize:
            max_fontsize = obj["size"]
    for obj in page.extract_words(use_text_flow=True, extra_attrs=["size"]):
        if abs(obj["size"] - max_fontsize) < 0.5:
            title_lines.append(obj["text"])
    return " ".join(title_lines).strip() if title_lines else "Không rõ tiêu đề"

def extract_authors_and_date(text):
    # Tìm ngày duyệt
    date_match = re.search(r"(Duyệt đăng|Ngày chấp nhận|Accepted).*?:?\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    approved_date = date_match.group(2) if date_match else "Không rõ ngày"
    # Tìm dòng chứa tên tác giả
    author_match = re.search(r"\n(.+?)\s*\*", text)
    authors = author_match.group(1).strip() if author_match else "Không rõ tác giả"
    authors = re.sub(r'[\d*]+', '', authors).strip()
    return authors, approved_date

def is_image_too_small(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.size[0] < 100 or image.size[1] < 100
    except Exception:
        return True

def is_black_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(gray < 30) / gray.size > 0.9

def is_white_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(gray > 245) / gray.size > 0.99

def check_and_repair_image(image_bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None, "decode_failed"
        if is_image_too_small(image_bytes):
            return None, "too_small"
        if is_black_image(image):
            return None, "too_black"
        if is_white_image(image):
            return None, "too_white"
        # ⚠️ Không tăng độ tương phản – giữ ảnh gốc
        repaired_img_bytes = cv2.imencode('.png', image)[1]
        return repaired_img_bytes.tobytes(), "viewable"
    except Exception:
        return None, "error"


def save_image(image_bytes, img_filename):
    try:
        with open(img_filename, "wb") as img_file:
            img_file.write(image_bytes)
        return True
    except Exception:
        return False

def extract_images_and_captions(pdf_path, output_folder):
    # conn = connect_to_mysql()
    # cursor = conn.cursor()
    total_extracted = 0
    viewable = 0
    extraction_date = datetime.now()
    try:
        doc = fitz.open(pdf_path)
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        real_pages = {}
        captions_by_page = {}
        doi = title = authors = approved_date = None

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and i == 0:
                    title = extract_title_by_fontsize(page)
                    authors, approved_date = extract_authors_and_date(text)
                    doi = extract_doi(text)
                bottom_texts = [
                    obj for obj in page.extract_words()
                    if obj["top"] > (page.height * 0.85)  # chỉ lấy những dòng nằm ở 15% cuối trang
                ]

                # tìm những dòng nằm gần giữa trang (xét theo toạ độ x)
                center_x = page.width / 2
                closest_to_center = None
                min_distance = float("inf")

                for word in bottom_texts:
                    word_center = (word["x0"] + word["x1"]) / 2
                    distance = abs(center_x - word_center)
                    if distance < min_distance and re.fullmatch(r"\d{1,3}", word["text"]):  # chỉ nhận số nguyên
                        min_distance = distance
                        closest_to_center = word["text"]

                if closest_to_center:
                    real_pages[i + 1] = int(closest_to_center)

                captions = [line.strip() for line in text.split('\n') if re.search(r'(hình|figure)\s*\d+', line.lower())]
                captions_by_page[i + 1] = captions

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                if page_num == 0 and img_index == 0:
                    continue  # bỏ ảnh đầu tiên

                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue
                repaired_image, status = check_and_repair_image(image_bytes)
                if status != "viewable":
                    continue

                os.makedirs(os.path.join(output_folder, 'extracted_images'), exist_ok=True)
                real_page = real_pages.get(page_num + 1, page_num + 1)
                img_name = f"{pdf_filename}_page_{real_page}_img_{img_index + 1}.png"
                img_path = os.path.join(output_folder, 'extracted_images', img_name)

                # Caption logic
                caption_list = captions_by_page.get(page_num + 1, [])
                caption = caption_list[img_index] if img_index < len(caption_list) else "Không có chú thích"
                caption_cleaned = re.sub(r'^(hình|hình ảnh|figure) \d+[.:]?', '', caption, flags=re.IGNORECASE).strip()

                # Lưu ảnh và vào DB
                viewable += 1
                # cursor.execute("""
                # INSERT INTO research (image_field_name, doi, title, caption, page_number, extraction_date, authors, approved_date, language)
                #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                # """, (
                #     img_name,
                #     doi or "No DOI found",
                #     title or "Không rõ tiêu đề",
                #     caption_cleaned,
                #     real_page,
                #     extraction_date,
                #     authors,
                #     approved_date,
                #     'vi'  # nếu là PDF tiếng Việt
                # ))


                save_image(repaired_image, img_path)
                total_extracted += 1
        # conn.commit()
    except Exception as e:
        print("Lỗi:", e)
    # finally:
        # cursor.close()
        # conn.close()
    return total_extracted, viewable, 0, time.time()

def select_folder(title):
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

# pdf_folder = select_folder("Chọn thư mục chứa PDF")
# output_folder = select_folder("Chọn thư mục lưu ảnh")
pdf_folder = r"E:\similarity_image\baibao"
output_folder = r"E:\similarity_image\pdf"
total_extracted = total_viewable = total_time = 0
for file in os.listdir(pdf_folder):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, file)
        print(f"\n🔍 Xử lý: {file}")
        start = time.time()
        extracted, viewable, _, _ = extract_images_and_captions(pdf_path, output_folder)
        end = time.time()
        total_extracted += extracted
        total_viewable += viewable
        total_time += (end - start)
        print(f"🕒 Thời gian: {end - start:.2f}s | 📸: {extracted} | ✅: {viewable}")

print("\n📊 Tổng kết")
print(f"🖼️ Tổng ảnh trích xuất: {total_extracted}")
print(f"✅ Tổng ảnh xem được : {total_viewable}")
print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
