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

# Lấy danh sách tất cả ImageFileName từ database
cursor.execute("SELECT image_field_name FROM research WHERE image_field_name IS NOT NULL")
db_filenames = set(row[0] for row in cursor.fetchall())

# Duyệt toàn bộ thư mục và lấy danh sách ảnh trong thư mục
all_image_files = []
for dirpath, _, filenames in os.walk(image_root_folder):
    for filename in filenames:
        all_image_files.append(filename)

# Tìm các file ảnh có trong thư mục nhưng không có trong database
extra_files = [f for f in all_image_files if f not in db_filenames]

# In kết quả
print("Các ảnh CÓ TRONG THƯ MỤC nhưng KHÔNG có trong database:")
for file in extra_files:
    print(file)

print(f"Tổng cộng {len(extra_files)} ảnh không có trong CSDL.")

# Đóng kết nối
cursor.close()
conn.close()
