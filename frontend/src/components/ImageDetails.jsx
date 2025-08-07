import React, { useEffect, useMemo, useState } from "react";
import Heading from "./Heading";
import PopupDetails from "./PopupDetails";
import { FaMagnifyingGlass } from "react-icons/fa6";
import StatisticsPanel from "./StatisticsPanel";
const ImageDetails = ({ similarity, metadata }) => {
  const [selectedClass, setSelectedClass] = useState(null); // Lưu lớp phân loại đang chọn
  const [showPopup, setShowPopup] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null); // Lưu thông tin ảnh đã chọn
  const [modelName, setModelName] = useState("");
  const [threshold, setThreshold] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [showStatisticPopup, setShowStatisticPopup] = useState(false);
  const [debouncedSearchItem, setDebounchSearchItem] = useState("");

  const classList = similarity?.flatMap((info) => info.predicted_class);
  const uniqueClasses = [...new Set(classList)];

  const filteredImages = similarity?.filter((info) =>
    selectedClass ? info.predicted_class === selectedClass : true
  );
  const handleShowImageDetail = (info, image) => {
    setSelectedImage({ info, image });
    setShowPopup(true);
  };

  console.log(similarity);

  useEffect(() => {
    if (similarity.length > 0) {
      let name = similarity[0].model_type;
      if (name.includes("vgg16")) {
        name = "VGG16";
      } else if (name.includes("convnext_v2")) {
        name = "ConvNeXt V2";
      } else {
        name = "AlexNet";
      }
      setModelName(name);
      setThreshold(similarity[0].threshold);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebounchSearchItem(searchTerm);
    }, 300);
    return () => clearTimeout(timer);
  });
  const highlightMatch = (text, keyword) => {
    if (!keyword) return text;

    const parts = text.split(new RegExp(`(${keyword})`, "gi"));
    return parts.map((part, i) =>
      part.toLowerCase() === keyword.toLowerCase() ? (
        <span
          key={i}
          className="text-red-500 font-semibold bg-yellow-100 px-1 rounded"
        >
          {part}
        </span>
      ) : (
        part
      )
    );
  };
  const searchedImages = useMemo(() => {
    if (!filteredImages) return [];

    const keyword = debouncedSearchItem.toLowerCase();
    return filteredImages?.map((info) => {
      const matchedImages = info?.similar_images?.filter(
        (img) =>
          img.title?.toLowerCase().includes(keyword) ||
          img.caption?.toLowerCase().includes(keyword) ||
          img.authors?.toLowerCase().includes(keyword)
      );
      return {
        ...info,
        matchedImages: matchedImages ?? [],
      };
    });
  }, [filteredImages, debouncedSearchItem]);

  return (
    <div className="  space-y-2 ">
      <div className="w-full my-4    ">
        <div className="bg-white border border-gray-300 rounded-lg     shadow  p-6 w-full">
          <div className="flex flex-wrap   gap-4  mb-6 w-full">
            <button
              onClick={() => setSelectedClass(null)}
              className={`px-3 py-2 rounded-full text-sm font-semibold transition-colors duration-300 ${
                !selectedClass
                  ? "bg-blue-600 text-white shadow-md"
                  : "text-blue-600 bg-transparent border border-blue-600 hover:bg-blue-50"
              }`}
            >
              All Classes
            </button>
            {uniqueClasses?.map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedClass(cls)}
                className={`px-3 py-2 rounded-full text-sm font-semibold transition-colors duration-300 ${
                  selectedClass === cls
                    ? "bg-blue-600 text-white shadow-md"
                    : "text-blue-600 bg-transparent border border-blue-600 hover:bg-blue-50"
                }`}
              >
                {cls}
              </button>
            ))}
          </div>
          <div className="flex flex-col md:flex-row items-center justify-between gap-2  w-full">
            <div className="relative w-full md:max-w-xl">
              <FaMagnifyingGlass className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-400" />
              <input
                type="text"
                placeholder="Search by title, caption, or authors"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-12 pr-4 py-2   rounded-xl border border-blue-300 bg-white
                     text-blue-900 placeholder-blue-400 focus:outline-none focus:ring-1 
                     focus:ring-blue-400 focus:border-blue-500 transition"
              />
            </div>
            <div>
              <div className="flex gap-8 text-blue-500 font-semibold text-sm md:text-base whitespace-nowrap">
                <div>
                  <span className="text-blue-500">Model:</span>{" "}
                  <span>{modelName}</span>
                </div>
                <div>
                  <span className="text-blue-600">Threshold:</span>{" "}
                  <span>{threshold}</span>
                </div>
              </div>
            </div>
          </div>
          <div
            onClick={() => setShowStatisticPopup(true)}
            className="text-end text-sm italic absolute right-14 cursor-pointer text-orange-500 hover:text-red-500 hover:underline  "
          >
            View statistis
          </div>
        </div>
      </div>

      {searchedImages?.map((info, index) => {
        const keyword = debouncedSearchItem.toLowerCase(); 
        const imagesToShow = info.matchedImages;

        return (
          <div
            key={index}
            className="grid grid-cols-1 md:grid-cols-[1fr_2.5fr] gap-5 bg-white shadow  rounded-lg  p-6 border border-gray-200"
          >
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
                  <span className="text-blue-600 font-semibold">
                    {info.predicted_class}
                  </span>{" "}
                  ({info.confidence})
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

            <div className="overflow-y-auto max-h-[600px] custom-scrollbar px-1">
              <h2 className="font-semibold text-blue-500 mb-3 border-b pb-1">
                {imagesToShow.length} Similar Images
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {imagesToShow.map((sim, simIdx) => (
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
                    <div className="mt-3 text-sm h-full text-gray-700 space-y-1 overflow-y-auto flex flex-col">
                      <p>
                        <strong>Confidence:</strong>{" "}
                        {(sim.similarity * 100).toFixed(2)}%
                      </p>
                      <p>
                        <strong>Title:</strong>{" "}
                        {highlightMatch(sim.title || "", keyword)}
                      </p>
                      <p>
                        <strong>Authors:</strong>{" "}
                        {highlightMatch(sim.authors || "N/A", keyword)}
                      </p>
                      <p className="">
                        <strong>Caption:</strong>{" "}
                        {highlightMatch(sim.caption || "N/A", keyword)}
                      </p>

                      {sim.doi && (
                        <a
                          href={`https://doi.org/${sim.doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          <strong>DOI:</strong>{" "}
                          {highlightMatch(sim.doi || "N/A", keyword)}
                        </a>
                      )}
                      <p
                        onClick={() => handleShowImageDetail(info, sim)}
                        className="flex items-center gap-1 text-emerald-500 hover:text-emerald-700 mt-auto self-end text-sm font-medium transition-all duration-200 cursor-pointer mb-1"
                      >
                        View detail
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}
      {showStatisticPopup && (
        <StatisticsPanel
          onClose={() => setShowStatisticPopup(false)}
          filteredImages={filteredImages}
          type={"image"}
        />
      )}

      {showPopup && (
        <PopupDetails
          originalImage={selectedImage?.info}
          similarImage={selectedImage?.image}
          metadata={metadata}
          onClose={() => setShowPopup(false)}
        />
      )}
    </div>
  );
};

export default ImageDetails;
