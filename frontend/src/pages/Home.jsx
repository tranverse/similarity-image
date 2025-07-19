import React from "react";
import Header from "@components/Header";
import { FaUpload, FaBrain, FaSearch } from "react-icons/fa";

const Home = () => {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 via-white to-rose-50 text-gray-800 font-sans">
      <Header />

      {/* Container chính căn giữa cả chiều dọc và ngang, max-width vừa phải */}
      <main className="flex flex-col flex-grow justify-center items-center px-6 max-w-7xl mx-auto">
        <section className="text-center mb-16 max-w-3xl">
          <h1 className="text-3xl md:text-4xl font-bold text-blue-500 drop-shadow-md mb-6">
            Similar Image Finder System
          </h1>
          <p className="text-base md:text-base text-gray-600 leading-relaxed tracking-wide">
            Discover visually similar images from your image or document using
            deep learning models including ConvNeXt V2, VGG16, and AlexNet.
            Upload an image or PDF and let AI efficiently find similar results
            within the scientific articles’ image database of Can Tho University
            Publishing House and the CTU Journal of Science.
          </p>
        </section>

        <section className="w-full grid grid-cols-1 md:grid-cols-3 gap-10 max-w-5xl">
          {/* Step 1 */}
          <div className="group bg-gradient-to-t from-white to-sky-100 rounded-3xl p-8 shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out cursor-default">
            <div className="flex justify-center text-blue-500 text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">
              <FaUpload />
            </div>
            <h3 className="text-xl font-semibold text-center text-blue-500 mb-3">
              1. Choose Image or PDF
            </h3>
            <p className="text-center text-gray-600 leading-relaxed tracking-wide">
              Upload <strong>either</strong> an image (PNG, JPG, JPEG){" "}
              <strong>or</strong> a PDF file. The system will extract images if
              it’s a PDF.
            </p>
          </div>

          {/* Step 2 */}
          <div className="group bg-gradient-to-t from-white to-yellow-100 rounded-3xl p-8 shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out cursor-default">
            <div className="flex justify-center text-yellow-600 text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">
              <FaBrain />
            </div>
            <h3 className="text-xl font-semibold text-center text-yellow-700 mb-3">
              2. Feature Extraction
            </h3>
            <p className="text-center text-gray-600 leading-relaxed tracking-wide">
              Our deep learning model extracts high-level features from the
              uploaded image(s).
            </p>
          </div>

          {/* Step 3 */}
          <div className="group bg-gradient-to-t from-white to-rose-100 rounded-3xl p-8 shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out cursor-default">
            <div className="flex justify-center text-rose-600 text-5xl mb-6 group-hover:scale-110 transition-transform duration-300">
              <FaSearch />
            </div>
            <h3 className="text-xl font-semibold text-center text-rose-700 mb-3">
              3. Find Similar Images
            </h3>
            <p className="text-center text-gray-600 leading-relaxed tracking-wide">
              The system compares features and shows you the most visually
              similar results.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Home;
