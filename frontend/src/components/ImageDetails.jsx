import React, { useState } from "react";
import Heading from "./Heading";
import PopupDetails from "./PopupDetails";
import { FaMagnifyingGlass } from "react-icons/fa6";

const ImageDetails = ({ similarity, metadata }) => {
  const [selectedClass, setSelectedClass] = useState(null); // Lưu lớp phân loại đang chọn
  const [showPopup, setShowPopup] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null); // Lưu thông tin ảnh đã chọn

  // Lọc tất cả các lớp phân loại có sẵn
  const classList = similarity?.flatMap((info) => info.predicted_class);
  const uniqueClasses = [...new Set(classList)]; // Loại bỏ trùng lặp

  // Lọc các ảnh tương đồng theo lớp phân loại đã chọn
  const filteredImages = similarity?.filter((info) =>
    selectedClass ? info.predicted_class === selectedClass : true
  );
  const handleShowImageDetail = (info, image) => {
    setSelectedImage({ info, image }); // Cập nhật ảnh khi người dùng chọn
    setShowPopup(true); // Hiển thị popup
    console.log(info, metadata);
  };
  return (
    <div className=" space-y-12">
      {/* Tabs chọn lớp */}
      <div className="flex flex-wrap gap-3 mb-2 justify-end">
        {uniqueClasses?.map((cls) => (
          <button
            key={cls}
            onClick={() => setSelectedClass(cls)}
            className={`px-4 py-2 rounded-full text-sm font-medium shadow transition-all duration-200 ${
              selectedClass === cls
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-700 hover:bg-blue-500 hover:text-white border border-blue-600"
            }`}
          >
            {cls}
          </button>
        ))}
        <button
          onClick={() => setSelectedClass(null)}
          className={`px-4 py-2 rounded-full text-sm font-medium shadow transition-all duration-200 ${
            !selectedClass
              ? "bg-indigo-600 text-white"
              : "bg-white text-gray-700 hover:bg-blue-500 hover:text-white border border-blue-600"
          }`}
        >
          All Classes
        </button>
      </div>

      {/* Hiển thị ảnh */}
      {filteredImages?.map((info, index) => (
        <div
          key={index}
          className="grid grid-cols-1 md:grid-cols-[1fr_2.5fr] gap-5 bg-white shadow-xl rounded-2xl p-6 border border-gray-200"
        >
          {/* Ảnh đầu vào */}
          <div className="bg-gray-50 p-4 rounded-xl shadow-inner border border-gray-100">
            <img
              src={`data:image/jpeg;base64,${info.image}`}
              alt="Input"
              className="w-full h-60 object-contain rounded-lg"
            />
            <div className="mt-5 text-sm text-gray-800 space-y-2">
              <h2 className="text-xl font-semibold text-center text-gray-800 mb-2 border-b pb-2">
                Input Image Info
              </h2>
              <p>
                <strong>Predicted Class:</strong>{" "}
                <span className="text-indigo-600 font-semibold">
                  {info.predicted_class}
                </span>{" "}
                ({info.confidence})
              </p>
              <p>
                <strong>Name:</strong> {info.name || "N/A"}
              </p>
              {info.doi && (
                <a
                  href={`https://doi.org/${info.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline block"
                >
                  DOI: {info.doi}
                </a>
              )}
              {info?.all_classes && (
                <div className="mt-4">
                  <h3 className="font-semibold text-gray-800 mb-1">
                    Other Class Probabilities:
                  </h3>
                  <ul className="list-disc list-inside text-gray-700 text-sm pl-2 space-y-1">
                    {info.all_classes
                      .sort((a, b) => b.confidence - a.confidence)
                      .slice(1)
                      .map((cls, idx) => (
                        <li key={idx}>
                          {cls.label}: {cls.confidence.toFixed(2)}%
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* Ảnh tương đồng */}
          <div className="overflow-y-auto max-h-[600px]  custom-scrollbar px-1">
            <h2 className="text-xl font-semibold text-gray-80 mb-3 border-b pb-1">
              {info?.similar_images?.length} Similar Images
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 ">
              {info?.similar_images?.map((sim, simIdx) => (
                <div
                  key={simIdx}
                  className="bg-white h-[450px] flex flex-col border border-gray-200 rounded-xl p-3 shadow-md hover:shadow-lg transition-transform hover:scale-[1.02]"
                >
                  <div className="h-[400px] flex justify-center items-center overflow-hidden">
                    <img
                      src={`http://127.0.0.1:8000/media/dataset/${info.predicted_class}/${sim.image_field_name}`}
                      alt={sim.image_field_name}
                      className="rounded-md w-full object-contain h-full bg-gray-200 p-1"
                    />
                  </div>
                  <div className="mt-3 text-sm h-full  text-gray-700 space-y-1 overflow-y-auto flex flex-col ">
                    <p>
                      <strong>Confidence:</strong>{" "}
                      {(sim.similarity * 100).toFixed(2)}%
                    </p>
                    <p>
                      <strong>Name:</strong> {sim.image_field_name}
                    </p>
                    <p>
                      <strong>Caption:</strong> {sim.caption || "N/A"}
                    </p>
                    <p>
                      <strong>Page Number:</strong> {sim.page_number || "N/A"}
                    </p>
                    {sim.doi && (
                      <a
                        href={`https://doi.org/${sim.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline"
                      >
                        <strong>DOI:</strong> {sim.doi}
                      </a>
                    )}
                    <p
                      onClick={() => handleShowImageDetail(info, sim)}
                      className="flex items-center gap-1 text-emerald-500 hover:text-emerald-700 mt-auto self-end
                       text-sm font-medium  transition-all duration-200 cursor-pointer mb-1  "
                    >
                      View detail
                      <FaMagnifyingGlass />
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}

      {/* Popup chi tiết ảnh tương đồng */}
      {showPopup && (
        <PopupDetails
          originalImage={selectedImage?.info}
          similarImage={selectedImage?.image}
          metadata = {metadata}
          onClose={() => setShowPopup(false)}
        />
      )}
    </div>
  );
};

export default ImageDetails;
