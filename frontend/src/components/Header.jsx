import React from "react";
import { PawPrint } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import Logo from "@assets/images/logo.png";
const Header = () => {
  const { pathname } = useLocation();

  return (
    <header className="w-full bg-white shadow">
      <div className="px-6  flex items-center justify-between relative">
        <Link to={"/"}>
          <div className="flex items-center gap-2">
            <div className="flex">
              <img src={Logo} className="w-[50px] h-[50px]" alt="" />
            </div>
            <span className="text-lg sm:text-xl font-semibold text-blue-500">
              PicExtractor
            </span>
          </div>
        </Link>

        <nav className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-4 text-sm sm:text-base">
          <Link
            onClick={() => sessionStorage.removeItem("extractedState")}
            to="/single-image"
            className={`px-4 py-2 rounded-md transition font-medium ${
              pathname === "/single-image"
                ? "text-blue-600 bg-blue-50  "
                : "text-gray-500 hover:text-blue-500 hover:bg-gray-100"
            }`}
          >
            Reverse Image Search
          </Link>

          <Link
            to="/pdf"
            className={`px-4 py-2 rounded-lg font-semibold transition duration-300 ${
              pathname === "/pdf" || pathname === "/classify"
                ? "text-blue-600 bg-blue-50  "
                : "text-gray-500 hover:text-blue-600 hover:bg-blue-50"
            }`}
          >
            Reverse PDF Search
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
