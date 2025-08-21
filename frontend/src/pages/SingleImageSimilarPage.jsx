import Breadcrumb from "@components/Breadcrumb";
import Header from "@components/Header";
import ExtractedService from "@services/Extracted.service";
import React, { useRef, useState } from "react";
import { BiImageAdd } from "react-icons/bi";
import { toast } from "react-toastify";
import { IoCloseOutline } from "react-icons/io5";
import { SlTrash } from "react-icons/sl";
import { FaMagnifyingGlass } from "react-icons/fa6";
import TopPlagiarizedDocs from "@components/StatisticsPanel";

const SingleImageSimilarPage = () => {
  const [selectedModel, setSelectedModel] = useState("vgg16_aug");
  const [threshold, setThreshold] = useState(0.5);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const imgRef = useRef(null);
  const [inputImage, setInputImage] = useState([]);
  const [similarImages, setSimilarImages] = useState([]);
  const [isShowDetail, setIsShowDetail] = useState(false);
  const [imageViewDetail, setImageViewDetail] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [showStatisticPopup, setShowStatisticPopup] = useState(false);
  const keyword = searchTerm.toLowerCase();

  const filteredImages = similarImages?.similar_images?.filter((img) => {
    return (
      img?.title?.toLowerCase().includes(keyword) ||
      img?.caption?.toLowerCase().includes(keyword) ||
      img?.authors?.toLowerCase().includes(keyword)
    );
  });

  const handleChooseImage = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      const base64 = await fileToBase64(file);
      setInputImage([{ base64: base64 }]);
    }
  };
  const handleDropImage = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImageUrl(URL.createObjectURL(file));
      setImageFile(file);
      const base64 = await fileToBase64(file);
      setInputImage([{ base64: base64 }]);
    }
  };
  const handleDragOver = (e) => {
    e.preventDefault();
  };
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        let base64 = reader.result;
        if (base64.includes(",")) {
          base64 = base64.split(",")[1];
        }
        resolve(base64);
      };
      reader.onerror = (error) => reject(error);
    });
  };

  const handleFileSimilarImages = async () => {
    setIsLoading(true);
    try {
      const response = await ExtractedService.classify(
        inputImage,
        selectedModel,
        threshold
      );
      if (response.results && response.results.length > 0) {
        setSimilarImages(response.results[0]);
      } else {
        toast.error("No similar images found");
        setSimilarImages([]);
      }
    } catch (error) {
      toast.error("Error: " + (error.message || error));
      setSimilarImages([]);
    } finally {
      setIsLoading(false);
    }
  };
  const handleShowDetail = (img) => {
    setImageViewDetail(img);
  };
  const handleReloadImage = () => {
    setImageUrl(null);
    setImageFile(null);
    setSimilarImages([]);
    setInputImage([]);
  };
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
  return (
    <div className="bg-gradient-to-br from-blue-50 via-white to-rose-50 min-h-screen ">
      <Header></Header>

      <div className="flex mx-5 gap-4 mt-10 ">
        <div className="w-1/3 flex flex-col h-[600px]  ">
          <div
            className="relative h-4/5 min-h-0 items-center shadow mb-4 bg-white justify-center cursor-pointer border rounded-xl p-4 border-gray-200"
            onClick={() => {
              imgRef.current.click();
            }}
            onDrop={handleDropImage}
            onDragOver={handleDragOver}
          >
            {!imageUrl ? (
              <div className="flex flex-col justify-center items-center border-blue-400 w-full h-full border-dashed border-2 rounded-xl">
                <BiImageAdd className="text-9xl text-blue-400" />
                <p className="text-gray-500 font-semibold">
                  Drop, pase or upload an image
                </p>
              </div>
            ) : (
              <>
                <div
                  className="flex  justify-end absolute right-0 px-2"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleReloadImage();
                  }}
                >
                  <SlTrash className="cursor-pointer text-2xl text-blue-500 " />
                </div>

                <div className="flex items-center justify-center w-full h-full">
                  <img
                    src={imageUrl}
                    className=" max-h-full max-w-full object-contain"
                  />
                </div>
              </>
            )}

            <input
              accept="image/*"
              type="file"
              onChange={(e) => handleChooseImage(e)}
              className="hidden"
              ref={imgRef}
            />
          </div>

          <div className=" p-3 bg-white  shadow border border-gray-100 rounded-lg">
            <div className="flex gap-4">
              <div className="w-1/2">
                <p className="text-gray-700 mb-1">Select model</p>
                <select
                  id="modelSelector"
                  onChange={(e) => setSelectedModel(e.target.value)}
                  value={selectedModel}
                  className=" w-full cursor-pointer rounded-md bg-white py-1.5 pr-2 pl-3 text-left appearance-none
                  text-gray-700 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2
                    focus:outline-indigo-600 sm:text-sm/6"
                  aria-haspopup="listbox"
                  aria-expanded="true"
                  aria-labelledby="listbox-label"
                >
                  {/* <option value="vgg16">VGG16</option> */}
                  <option value="vgg16_aug">VGG16 </option>
                  {/* <option value="convnext_v2">ConvNeXt V2</option> */}
                  <option value="convnext_v2_aug">ConvNeXt V2 </option>
                  {/* <option value="alexnet">AlexNet</option> */}
                  <option value="alexnet_aug">AlexNet </option>
                </select>
              </div>
              <div className="w-1/2 ">
                <p className="text-gray-700 mb-1">Select threshold</p>
                <select
                  id="similarity"
                  name="similarity"
                  onChange={(e) => setThreshold(e.target.value)}
                  value={threshold}
                  className="w-full appearance-none outline-1 rounded-md sm:text-sm/6 pr-16 focus:outline-2 focus:outline-indigo-500 focus:-outline-offset-2  pl-3 cursor-pointer py-1.5 outline-gray-300 -outline-offset-1"
                >
                  {[...Array(10)].map((_, i) => {
                    const val = ((i + 1) / 10).toFixed(1);
                    return (
                      <option key={val} value={val}>
                        {val}
                      </option>
                    );
                  })}
                </select>
              </div>
            </div>

            <div
              className=" justify-center mt-2"
              onClick={handleFileSimilarImages}
            >
              <button
                disabled={imageUrl === null || isLoading === true}
                className={`bg-blue-400 text-white font-semibold py-2 px-8 rounded-md   w-full   
                      hover:bg-blue-500 transition duration-300 ease-in-out ${
                        isLoading === true || imageUrl === null
                          ? "cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
              >
                Find Similar Images
              </button>
            </div>
          </div>
        </div>

        {isShowDetail ? (
          <div className="w-2/3 flex flex-col h-[600px] bg-white border border-gray-200 rounded-xl shadow-md">
            <div className="flex justify-between items-center pl-4 py-1 border-b border-gray-200">
              <div className="text-center text-blue-500 select-none">
                <span>
                  <strong>Predicted class:</strong>{" "}
                  {similarImages?.predicted_class || "N/A"}
                </span>
                <span className="mx-2 text-blue-400">|</span>
                <span>
                  <strong>Confidence:</strong>{" "}
                  {similarImages?.confidence || "N/A"}
                </span>
              </div>
              <button
                onClick={() => setIsShowDetail(false)}
                className="cursor-pointer text-2xl text-blue-700 hover:text-white hover:bg-blue-500 rounded-full p-1 transition"
                aria-label="Close"
                title="Close"
              >
                <IoCloseOutline />
              </button>
            </div>

            <div className="flex flex-grow h-[550px]">
              <div className="w-3/5 flex items-center justify-center p-4 bg-gray-50   ">
                <img
                  src={`http://127.0.0.1:8000/media/dataset/${similarImages?.predicted_class}/${imageViewDetail?.image_field_name}`}
                  alt="Image detail"
                  className="object-contain w-full max-h-full "
                />
              </div>

              <div className="w-2/5 p-6 text-gray-700 text-sm overflow-y-auto space-y-4">
                <p>
                  <span className="font-semibold">Similarity:</span>{" "}
                  {imageViewDetail?.similarity != null
                    ? (imageViewDetail.similarity * 100).toFixed(2) + "%"
                    : "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Title:</span>{" "}
                  {imageViewDetail?.title || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Authors:</span>{" "}
                  {imageViewDetail?.authors || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Caption:</span>{" "}
                  {imageViewDetail?.caption || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Page Number:</span>{" "}
                  {imageViewDetail?.page_number || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Accepted Date:</span>{" "}
                  {imageViewDetail?.approved_date || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">DOI:</span>{" "}
                  <a
                    href={`https://doi.org/${imageViewDetail?.doi}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {imageViewDetail?.doi}
                  </a>
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="w-2/3 flex flex-col  bg-white  shadow border border-gray-100 rounded-lg h-[600px] p-2">
            <div className="border-b border-blue-400 px-4 py-2 flex flex-col gap-2   ">
              {similarImages?.similar_images?.length >= 0 ? (
                <>
                  <div className="flex justify-between items-center">
                    <p className="text-blue-500 font-semibold">
                      {filteredImages.length} Similar Images
                    </p>

                    <div className="text-blue-500 ">
                      <strong className="font-semibold">
                        Predicted class:
                      </strong>{" "}
                      {similarImages?.predicted_class || "N/A"}{" "}
                      <span className="mx-2 text-blue-400">|</span>
                      <strong className="font-semibold">
                        Confidence:
                      </strong>{" "}
                      {similarImages?.confidence || "N/A"}
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="relative w-md      ">
                      <FaMagnifyingGlass className="absolute left-4 top-1/2 -translate-y-1/2 text-blue-400" />
                      <input
                        type="text"
                        placeholder="Search by title, caption, or authors"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-12 pr-4 py-1   rounded-xl border border-blue-300 bg-white
                                 text-blue-900 placeholder-blue-400 focus:outline-none focus:ring-1 
                                 focus:ring-blue-400 focus:border-blue-500 transition"
                      />
                    </div>
                    <div
                      onClick={() => setShowStatisticPopup(true)}
                      className="text-end text-sm italic  cursor-pointer text-orange-500 hover:text-red-500 hover:underline  "
                    >
                      View statistis
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-blue-500 font-semibold text-center">
                  This section displays results from the similar image retrieval
                  function
                </p>
              )}
            </div>

            {isLoading ? (
              <div className="flex-grow flex flex-col items-center justify-center gap-2 ">
                <span className=" loading loading-infinity loading-xl  scale-200 bg-gradient-to-r from-blue-300 to-blue-700"></span>
                <p className="font-semibold text-lg text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-blue-700">
                  Wait for the system to extract images
                </p>
              </div>
            ) : (
              <div
                className={`grid grid-cols-1 sm:grid-cols-2  gap-2 mt-2  h-full overflow-y-auto 
                  ${
                    similarImages?.similar_images?.length === 0
                      ? "md:grid-cols-1"
                      : "md:grid-cols-2"
                  } `}
              >
                {similarImages?.similar_images?.length === 0 && (
                  <div className="flex flex-col items-center justify-center flex-grow h-full w-full text-gray-500">
                    <p className="text-lg font-medium">
                      No similar images found
                    </p>
                  </div>
                )}

                {filteredImages?.map((img, index) => (
                  <div
                    className="h-[520px] flex flex-col border border-gray-200 rounded-xl p-4 shadow-md hover:shadow-lg transition-shadow duration-300 bg-white space-y-3"
                    key={index}
                  >
                    <div className="flex justify-center items-center h-[450px] overflow-hidden ">
                      <img
                        src={`http://127.0.0.1:8000/media/dataset/${similarImages?.predicted_class}/${img?.image_field_name}`}
                        className="w-full h-full p-2 bg-gray-100 object-contain rounded-md"
                        alt="Similar"
                      />
                    </div>

                    <div className="flex flex-col space-y-1 text-sm text-gray-700 h-full overflow-y-auto ">
                      <p>
                        <span className="font-semibold text-gray-800">
                          Similarity:
                        </span>{" "}
                        {(img?.similarity * 100).toFixed(2) || "N/A"}%
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          Title:
                        </span>{" "}
                        {highlightMatch(img?.title || "N/A", keyword)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          Authors:
                        </span>{" "}
                        {highlightMatch(img?.authors || "N/A", keyword)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          Caption:
                        </span>{" "}
                        {highlightMatch(img?.caption || "N/A", keyword)}
                      </p>

                      <p>
                        <span className="font-semibold text-gray-800">
                          DOI:
                        </span>{" "}
                        <a
                          href={`https://doi.org/${img?.doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {img?.doi}
                        </a>
                      </p>
                      <p
                        onClick={() => {
                          handleShowDetail(img);
                          setIsShowDetail(true);
                        }}
                        className="cursor-pointer text-green-400 hover:text-green-700 mt-auto self-end"
                      >
                        <span className="font-semibold">View detail</span>
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {showStatisticPopup && (
          <TopPlagiarizedDocs
            onClose={() => setShowStatisticPopup(false)}
            filteredImages={similarImages}
          />
        )}
      </div>
    </div>
  );
};

export default SingleImageSimilarPage;
