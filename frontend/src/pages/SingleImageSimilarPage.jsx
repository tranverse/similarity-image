import Breadcrumb from "@components/Breadcrumb";
import Header from "@components/Header";
import ExtractedService from "@services/Extracted.service";
import React, { useRef, useState } from "react";
import { BiImageAdd } from "react-icons/bi";
import { toast } from "react-toastify";
import { IoCloseOutline } from "react-icons/io5";

const SingleImageSimilarPage = () => {
  const [selectedModel, setSelectedModel] = useState("vgg16_aug");
  const [threshold, setThreshold] = useState(0);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const imgRef = useRef(null);
  const [inputImage, setInputImage] = useState([]);
  const [similarImages, setSimilarImages] = useState([]);
  const [isShowDetail, setIsShowDetail] = useState(false);
  const [imageViewDetail, setImageViewDetail] = useState({});
  const handleChooseImage = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      const base64 = await fileToBase64(file);
      setInputImage([{ base64: base64 }]);
    }
  };
  const handleDropImage = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setImageUrl(URL.createObjectURL(file));
      setImageFile(file);
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
  // console.log(inputImage);

  const handleFileSimilarImages = async () => {
    const response = await ExtractedService.classify(
      inputImage,
      selectedModel,
      threshold
    );

    try {
      setSimilarImages(response.results[0]);
    } catch (error) {
      toast.error("Error: ", error);
    }
  };
  const handleShowDetail = (img) => {
    setImageViewDetail(img);
  };
  return (
    <div>
      <Header></Header>
      {/* <Breadcrumb
        items={[{ label: "Extract Images", to: "/", icon: "" }]}
      ></Breadcrumb> */}
      <div className="flex mx-5 gap-4 mt-12">
        <div className="w-1/3 flex flex-col h-[600px] ">
          <div
            className=" h-4/5 min-h-0 items-center shadow mb-4  justify-center cursor-pointer border rounded-xl p-4  border-gray-200"
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
              <div className="flex items-center justify-center w-full h-full">
                <img
                  src={imageUrl}
                  className=" max-h-full max-w-full object-contain"
                />
              </div>
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
              className=" justify-center mt-2 "
              onClick={handleFileSimilarImages}
            >
              <button
                disabled={imageUrl === null}
                className={`bg-blue-500 text-white font-semibold py-2 px-8 rounded-md   w-full 
                      hover:bg-blue-400 transition duration-300 ease-in-out`}
              >
                Find Similar Images
              </button>
            </div>
          </div>
        </div>  

        {isShowDetail ? (
          <div className="w-2/3 flex flex-col h-[600px] bg-white border border-gray-100 rounded-lg shadow ">
            <div className="flex justify-between px-2 pt-1 ">
              <div className="text-center text-blue-700 font-medium mb-2">
                <span>
                  <strong>Predicted class:</strong>{" "}
                  {similarImages?.predicted_class || "N/A"}
                </span>
                <span className="mx-1 text-blue-400">|</span>
                <span>
                  <strong>Confidence:</strong>{" "}
                  {similarImages?.confidence || "N/A"}
                </span>
              </div>
              <button
                onClick={() => setIsShowDetail(false)}
                className="cursor-pointer text-3xl text-gray-600 hover:text-red-500 transition"
                aria-label="Close"
              >
                <IoCloseOutline />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr]  ">
              <div className=" flex flex-1  items-center h-[500px] justify-center p-2   ">
                <img
                  src={`http://127.0.0.1:8000/media/dataset/${similarImages?.predicted_class}/${imageViewDetail?.image_field_name}`}
                  alt="Image detail" 
                  className="object-contain w-full h-full "
                />
              </div>

              <div className="flex flex-col text-sm text-gray-700 p-3 overflow-y-auto space-y-2">

                <p>
                  <span className="font-semibold">Similarity:</span>{" "}
                  {(imageViewDetail?.similarity * 100).toFixed(2) || "N/A"}%
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
                  <span className="font-semibold">Page Number:</span>{" "}
                  {imageViewDetail?.page_number || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">Accepted Date:</span>{" "}
                  {imageViewDetail?.approved_date || "N/A"}
                </p>
                <p>
                  <span className="font-semibold">DOI:</span>{" "}
                  <span className="break-words text-blue-600 hover:underline">
                    {imageViewDetail?.doi || "N/A"}
                  </span>
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="w-2/3 flex flex-col bg-white  shadow border border-gray-100 rounded-lg h-[600px] p-2">
            <div className="border-b border-blue-400 px-4 py-2 flex justify-between items-center">
              {similarImages?.similar_images?.length >= 0 ? (
                <>
                  <p className="text-blue-700 font-semibold">
                    {similarImages?.similar_images?.length ?? 0} Similar Images
                  </p>

                  <div className="text-blue-600 ">
                    <strong className="font-semibold">Predicted class:</strong>{" "}
                    {similarImages?.predicted_class || "N/A"}{" "}
                    <span className="mx-2 text-blue-400">|</span>
                    <strong className="font-semibold">Confidence:</strong>{" "}
                    {similarImages?.confidence || "N/A"}
                  </div>
                </>
              ) : (
                <p className="text-blue-500 font-semibold text-center">
                  This section displays results from the similar image retrieval
                  function
                </p>
              )}
            </div>

            <div
              className={`grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 gap-2 mt-2  h-full overflow-y-auto  `}
            >
              {similarImages?.similar_images?.length > 0 &&
                similarImages?.similar_images?.map((img, index) => (
                  <div
                    className="h-[550px] flex flex-col border border-gray-200 rounded-xl p-4 shadow-md hover:shadow-lg transition-shadow duration-300 bg-white space-y-3"
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
                        {img?.title || "N/A"}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          Caption:
                        </span>{" "}
                        {img?.caption || "N/A"}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          Page Number:
                        </span>{" "}
                        {img?.page_number || "N/A"}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-800">
                          DOI:
                        </span>{" "}
                        <span className="text-blue-600 hover:underline break-all">
                          {img?.doi || "N/A"}
                        </span>
                      </p>
                      <p
                        onClick={() => {
                          setIsShowDetail(!isShowDetail);
                          handleShowDetail(img);
                        }}
                        className="  cursor-pointer text-green-400 hover:text-green-700 mt-auto self-end   "
                      >
                        <span className="font-semibold ">View detail</span>
                      </p>
                    </div>
                  </div>
                ))}
              {similarImages?.similar_images?.length === 0 && (
                <div>
                  <p>No similar images </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SingleImageSimilarPage;
