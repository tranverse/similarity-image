// import React from "react";
// import Heading from "./Heading";
// const ImageDetails = ({ similarity }) => {
//   return (

//     <div className="p-6 space-y-12 bg-gradient-to-b from-gray-50 to-white min-h-screen ">
//               <Heading message=" Explore Visually Similar Images After Classification" />

//       {similarity?.map((info, index) => (
//         <div
//           key={index}
//           className="grid grid-cols-1 md:grid-cols-[1fr_3fr] gap-10 bg-white shadow-xl rounded-2xl p-6 border border-gray-200 h-[500px] overflow-auto"
//         >
//           {/* Ảnh đầu vào */}
//           <div className="bg-white p-5 rounded-xl shadow-inner border border-gray-100">
//             <img
//               src={`data:image/jpeg;base64,${info.image}`}
//               alt="Input"
//               className="w-full h-auto object-cover rounded-lg"
//             />
//             <div className="mt-5 text-sm text-gray-700 space-y-2">
//               <h2 className="text-xl font-bold text-center text-gray-800">
//                 Input Image Info
//               </h2>
//               <p className="text-base font-semibold text-gray-800">
//                 <span className="text-gray-700">Predicted Class:</span>{" "}
//                 <span className="text-indigo-600">{info.predicted_class}</span>{" "}
//                 <span className="text-indigo-600"> - {info.confidence}</span>
//               </p>

//               <p>
//                 <strong>Name:</strong> {info.name || "N/A"}
//               </p>
//               <p>
//                 <strong>Caption:</strong> {info.caption || "N/A"}
//               </p>
//               {info.doi && (
//                 <a
//                   href={`https://doi.org/${info.doi}`}
//                   target="_blank"
//                   rel="noopener noreferrer"
//                   className="block text-blue-600 hover:underline"
//                 >
//                   DOI: {info.doi}
//                 </a>
//               )}

//               {/* Hiển thị tất cả các lớp theo độ chính xác giảm */}
//               {info?.all_classes && (
//                 <div className="mt-4">
//                   <h3 className="font-semibold text-gray-800 mb-2">
//                     Other Class Probabilities:
//                   </h3>
//                   <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
//                     {info.all_classes
//                       .sort((a, b) => b.confidence - a.confidence)
//                       .slice(1) // Bỏ phần tử đầu tiên (cao nhất)
//                       .map((cls, idx) => (
//                         <li key={idx}>
//                           {cls.label}: {(cls?.confidence*100).toFixed(2)}%
//                         </li>
//                       ))}
//                   </ul>
//                 </div>
//               )}
//             </div>
//           </div>

//           {/* Ảnh tương đồng */}
//           <div className="overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
//             <h2 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">
//               Similar Images
//             </h2>
//             <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
//               {info?.similar_images?.map((sim, simIdx) => (
//                 <div
//                   key={simIdx}
//                   className="bg-white border border-gray-100 rounded-xl p-4 shadow-lg transition-transform hover:scale-[1.02]"
//                 >
//                   <img
//                     src={`http://127.0.0.1:8000/media/raw/${info.predicted_class}/${sim.image_field_name}`}
//                     alt={sim.image_field_name}
//                     className="rounded-lg w-full h-40 object-cover mb-3"
//                   />
//                   <div className="text-sm text-gray-700 space-y-1">
//                     <p>
//                       <strong>Confidence:</strong>{" "}
//                       {(sim.similarity * 100).toFixed(2)}%
//                     </p>
//                     <p>
//                       <strong>Name:</strong> {sim.image_field_name}
//                     </p>
//                     <p>
//                       <strong>Caption:</strong> {sim.caption || "N/A"}
//                     </p>
//                     {sim.doi && (
//                       <a
//                         href={`https://doi.org/${sim.doi}`}
//                         target="_blank"
//                         rel="noopener noreferrer"
//                         className="text-blue-600 hover:text-blue-900"
//                       >
//                         DOI: {sim.doi}
//                       </a>
//                     )}
//                     <p className="text-sm text-center text-blue-600 font-medium  hover:text-blue-800 cursor-pointer transition duration-200">
//                       View more details
//                     </p>
//                   </div>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>
//       ))}
//     </div>
//   );
// };

// export default ImageDetails;
import React, { useState } from "react";
import Heading from "./Heading";
import PopupDetails from "./PopupDetails";
const ImageDetails = ({ similarity }) => {
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
    setSelectedImage({info, image}); // Cập nhật ảnh khi người dùng chọn
    setShowPopup(true); // Hiển thị popup
    console.log(info, image);
  };
  return (
    <div className="p-6 space-y-12 bg-gradient-to-b from-gray-50 to-white min-h-screen">
      <Heading message="Explore Visually Similar Images After Classification" />

      {/* Tab để chọn lớp phân loại */}
      <div className="flex space-x-4 mb-6">
        {uniqueClasses?.map((cls) => (
          <button
            key={cls}
            onClick={() => setSelectedClass(cls)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedClass === cls
                ? "bg-blue-600 text-white"
                : "bg-white text-gray-700 hover:bg-blue-500 hover:text-white border border-blue-600"
            }`}
          >
            {cls}
          </button>
        ))}
        <button
          onClick={() => setSelectedClass(null)} // Hiển thị tất cả khi không chọn lớp
          className={`px-4 py-2 rounded-lg transition-colors ${
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
          className="grid grid-cols-1 md:grid-cols-[1fr_3fr] gap-10 bg-white shadow-xl rounded-2xl p-6 border border-gray-200 h-[500px] overflow-auto"
        >
          {/* Ảnh đầu vào */}
          <div className="bg-white p-5 rounded-xl shadow-inner border border-gray-100">
            <img
              src={`data:image/jpeg;base64,${info.image}`}
              alt="Input"
              className="w-full h-50 object-cover rounded-lg"
            />
            <div className="mt-5 text-sm text-gray-700 space-y-2">
              <h2 className="text-xl font-bold text-center text-gray-800">
                Input Image Info
              </h2>
              <p className="text-base font-semibold text-gray-800">
                <span className="text-gray-700">Predicted Class:</span>{" "}
                <span className="text-indigo-600">{info.predicted_class}</span>{" "}
                <span className="text-indigo-600"> - {info.confidence}</span>
              </p>

              <p>
                <strong>Name:</strong> {info.name || "N/A"}
              </p>
              {/* <p>
                <strong>Caption:</strong> {info.caption || "N/A"}
              </p> */}
              {info.doi && (
                <a
                  href={`https://doi.org/${info.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-blue-600 hover:underline"
                >
                  DOI: {info.doi}
                </a>
              )}
              {info?.all_classes && (
                <div className="mt-4">
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Other Class Probabilities:
                  </h3>
                  <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                    {info.all_classes
                      .sort((a, b) => b.confidence - a.confidence)
                      .slice(1) // Bỏ phần tử đầu tiên (cao nhất)
                      .map((cls, idx) => (
                        <li key={idx}>
                          {cls.label}: {(cls?.confidence).toFixed(2)}%
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* Ảnh tương đồng */}
          <div className="overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
            <h2 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">
              Similar Images
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
              {info?.similar_images?.map((sim, simIdx) => (
                <div
                  key={simIdx}
                  className="bg-white flex flex-col border border-gray-100 rounded-xl p-2 shadow-lg transition-transform hover:scale-[1.02]"
                >
                  <div className="h-[200px] flex justify-center items-center overflow-hidden">
                    <img
                      src={`http://127.0.0.1:8000/media/raw/${info.predicted_class}/${sim.image_field_name}`}
                      alt={sim.image_field_name}
                      className="rounded-lg w-full "
                    />
                  </div>
                  <div className="text-sm text-gray-700 space-y-1 flex-1">
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
                    {sim.doi && (
                      <a
                        href={`https://doi.org/${sim.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-900"
                      >
                        DOI: {sim.doi}
                      </a>
                    )}
                    <p className="cursor-pointer" onClick={() => handleShowImageDetail(info, sim)}>
                      View detail
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
      {showPopup  && (
        <PopupDetails
          originalImage={selectedImage?.info}
          similarImage={selectedImage?.image}
              onClose={() => setShowPopup(false)} // ✅ Truyền đúng hàm

        />
      )}
    </div>
  );
};

export default ImageDetails;
