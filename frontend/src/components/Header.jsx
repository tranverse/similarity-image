import React from 'react';
import { PawPrint } from 'lucide-react';

const Header = () => {
  return (
    <header className="w-full bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-md ">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo + Tên App */}
        <div className="flex items-center gap-3">
          <PawPrint className="w-8 h-8 text-white" />
          <span className="text-2xl font-bold tracking-wide">Classifier System</span>
        </div>

        {/* Điều hướng đơn giản */}
        {/* <nav className="hidden md:flex gap-6 text-sm font-medium">
          <a href="/" className="hover:text-gray-200 transition">Trang chủ</a>
          <a href="/about" className="hover:text-gray-200 transition">Giới thiệu</a>
          <a href="/upload" className="hover:text-gray-200 transition">Phân loại ảnh</a>
          <a href="/contact" className="hover:text-gray-200 transition">Liên hệ</a>
        </nav> */}
      </div>
    </header>
  );
};

export default Header;
