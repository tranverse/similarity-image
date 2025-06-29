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
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True

def connect_to_mysql():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='123456',
        database='bb2110065'
    )

def extract_doi(text):
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.IGNORECASE)
    return match.group(0) if match else None

def extract_title_by_fontsize(page):
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return "Không rõ tiêu đề"
    max_font = max(w['size'] for w in words)
    lines = [w['text'] for w in words if abs(w['size'] - max_font) < 0.5]
    return " ".join(lines).strip() or "Không rõ tiêu đề"

def extract_authors_and_date(text):
    date_match = re.search(r"(Accepted|Ngày chấp nhận|Duyệt đăng).*?:?\s*(\d{1,2} [A-Z][a-z]+ \d{4}|\d{2}/\d{2}/\d{4})", text)
    approved_date = date_match.group(2) if date_match else "Không rõ ngày"
    authors = "Không rõ tác giả"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "Author(s):" in line:
            authors = line.split(":", 1)[-1].strip()
            break
        if "doi" in line.lower() and i + 2 < len(lines):
            next_line = lines[i + 2].strip()
            if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", next_line):
                authors = next_line.strip()
                break
    # ✅ Xóa dấu * và số mũ nếu có
    authors = re.sub(r'[\*\d⁰¹²³⁴⁵⁶⁷⁸⁹]+', '', authors).strip()
    return authors, approved_date

def is_image_too_small(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.size[0] < 100 or img.size[1] < 100
    except:
        return True

def is_black_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (np.sum(gray < 30) / gray.size) > 0.9

def is_white_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (np.sum(gray > 245) / gray.size) > 0.99

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 80

def check_and_filter_image(image_bytes):
    try:
        img_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # if image is None or image.size == 0:
        #     return None, "decode_failed"
        # if is_image_too_small(image_bytes):
        #     return None, "too_small"
        if is_black_image(image):
            return None, "too_black"
        # if is_white_image(image):
        #     return None, "too_white"
        # if is_blurry(image):
        #     return None, "too_blurry"
        return image_bytes, "viewable"
    except:
        return None, "error"

def save_image(image_bytes, path):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(path, format="PNG")
        return True
    except:
        return False

def extract_images_and_captions(pdf_path, output_folder):
    start = time.time()
    # conn = connect_to_mysql()
    # cursor = conn.cursor()
    viewable_images = 0
    total_extracted = 0
    extraction_time = datetime.now()
    seen_hashes = set()

    try:
        doc = fitz.open(pdf_path)
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        doi = title = authors = approved_date = "Không rõ"
        real_pages = {}
        captions_by_page = {}

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
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
                if i == 0:
                    title = extract_title_by_fontsize(page)
                    authors, approved_date = extract_authors_and_date(text)
                    doi = extract_doi(text) or "No DOI"
        image_count_by_page = {}

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            text_blocks = page.get_text("dict")["blocks"]
            captions = []
            for blk in sorted(text_blocks, key=lambda b: b.get("bbox", [0])[1]):
                if blk.get("type") == 0:
                    text = " ".join(span["text"] for line in blk["lines"] for span in line["spans"])
                    if re.search(r'(figure|hình)\s*\d+', text.lower()):
                        captions.append(text.strip())
            captions_by_page[page_num + 1] = captions

            for img_index, img in enumerate(images):
                if page_num == 0 and img_index == 0:
                    continue
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue

                # try:
                #     hash_val = str(imagehash.phash(Image.open(io.BytesIO(image_bytes)).convert("RGB")))
                #     if hash_val in seen_hashes:
                #         continue
                #     seen_hashes.add(hash_val)
                # except:
                #     continue
                try:
                    Image.open(io.BytesIO(image_bytes)).verify()  # chỉ kiểm tra xem ảnh có lỗi không
                except Exception as e:
                    print(f"⚠️ Trang {page_num + 1} - ảnh #{img_index + 1}: lỗi ảnh ({e})")
                    continue


                repaired, status = check_and_filter_image(image_bytes)
                if status != "viewable":
                    continue

                real_page = real_pages.get(page_num + 1, page_num + 1)
                folder = os.path.join(output_folder, "extracted_images")
                os.makedirs(folder, exist_ok=True)
                # filename = f"{pdf_filename}_page_{real_page}_img_{img_index + 1}.png"
                real_page = real_pages.get(page_num + 1, page_num + 1)
                image_count_by_page.setdefault(real_page, 0)
                image_count_by_page[real_page] += 1
                image_number = image_count_by_page[real_page]

                filename = f"{pdf_filename}_page_{real_page}_img_{image_number}.png"
                print(f"🖼️  Trang {page_num + 1} - ảnh #{img_index + 1}:", end=" ")
                # if hash_val in seen_hashes:
                #     print("❌ Bỏ qua vì trùng hash")
                # elif status != "viewable":
                #     print(f"❌ Bỏ qua vì không xem được ({status})")
                # else:
                #     print("✅ Lưu thành công:", filename)

                path = os.path.join(folder, filename)

                caption_list = captions_by_page.get(page_num + 1, [])
                caption = caption_list[img_index] if img_index < len(caption_list) else "Không có chú thích"
                caption_clean = re.sub(r'^(figure|hình)\s*\d+[.:]?', '', caption, flags=re.IGNORECASE).strip()

                print(f"\n doi: {doi} \n title: {title}\n date: {approved_date}\n caption: {caption}\n authors: {authors}\n")
                # cursor.execute("""
                #     INSERT INTO research (image_field_name, doi, title, caption, page_number, extraction_date, authors, approved_date, language)
                #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                # """, (
                #     filename,
                #     doi or "No DOI found",
                #     title or "Không rõ tiêu đề",
                #     caption_clean,
                #     real_page,
                #     extraction_time,
                #     authors,
                #     approved_date,
                #     'en'
                # ))

                save_image(repaired, path)
                total_extracted += 1
                viewable_images += 1

        # conn.commit()
    except Exception as e:
        print("❌ Error:", e)
    # finally:
    #     cursor.close()
    #     conn.close()

    print(f"✅ Done: {total_extracted} ảnh | Xem được: {viewable_images} | Time: {time.time() - start:.2f}s")

def select_folder(title):
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

# pdf_folder = select_folder("📂 Chọn thư mục chứa PDF tiếng Anh")
# output_folder = select_folder("💾 Nơi lưu hình ảnh đã trích xuất")
pdf_folder = r'E:\similarity_image\baibao'
output_folder = r'E:\similarity_image\anhmoi'

total_extracted, total_viewable, total_time = 0, 0, 0
for file in os.listdir(pdf_folder):
    if file.lower().endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        extract_images_and_captions(path, output_folder)
