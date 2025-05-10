# print_structure.py
import os

def print_tree(startpath=r'E:\similarity_image\index.py', prefix=''):
    for item in sorted(os.listdir(startpath)):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{prefix}|-- {item}/")
            print_tree(path, prefix + "|   ")
        else:
            print(f"{prefix}|-- {item}")


if __name__ == '__main__':
    print_tree('.')  # In toàn bộ dự án từ thư mục hiện tại
