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

      </div>
    </header>
  );
};

export default Header;
