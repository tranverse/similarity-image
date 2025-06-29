import mysql.connector
import os

# Cấu hình kết nối cơ sở dữ liệu
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'luanvan'
}

# Thư mục gốc chứa ảnh (và các thư mục con)
image_root_folder = r'E:\phan_loai_goc'  # VD: "/home/user/images/"

# Hàm kiểm tra xem file có tồn tại trong bất kỳ thư mục con nào không
def file_exists_in_subfolders(filename, root_folder):
    for dirpath, _, files in os.walk(root_folder):
        if filename in files:
            return True
    return False

# Kết nối MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Truy vấn tất cả ImageID và ImageFileName
cursor.execute("SELECT image_id, image_field_name FROM research")
rows = cursor.fetchall()

deleted_count = 0

for image_id, image_file in rows:
    if image_file is None:
        continue  # Bỏ qua nếu không có tên file

    if not file_exists_in_subfolders(image_file, image_root_folder):
        print(f"Không tìm thấy ảnh: {image_file} -> Xóa bản ghi ID {image_id}")
        cursor.execute("DELETE FROM research WHERE image_id = %s", (image_id,))
        deleted_count += 1

# Lưu thay đổi và đóng kết nối
conn.commit()
cursor.close()
conn.close()

print(f"Đã xóa {deleted_count} bản ghi không có ảnh trong thư mục.")
