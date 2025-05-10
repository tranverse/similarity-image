import React, { useRef, useState } from "react";
import Button from "./Button";
import Photograph from "@assets/images/01-5689-A-VAN HUU HUE-1-10_page_2_img_1.png";
import ExtractedService from "@services/Extracted.service";
import { Link, useNavigate } from "react-router-dom";
import { IoReload } from "react-icons/io5";
import { IoIosExpand } from "react-icons/io";
const ExtractedImages = () => {
  const [pdfUrl, setPdfUrl] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const filePdfRef = useRef(null);
  const [isExtractedImages, setIsExtractedImages] = useState(false);
  const [images, setImages] = useState([]);
  const [selectedModel, setSelectedModel] = useState("vgg16");
  const [threshold, setThreshold] = useState(0);

  const navigate = useNavigate();
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type == "application/pdf") {
      const url = URL.createObjectURL(file);
      setPdfUrl(url);
      setPdfFile(file);
    }
  };
  const handleExtractImages = async () => {
    const images = await ExtractedService.sendPdf({ pdf: pdfFile });
    setImages(images);
    if (images) {
      setIsExtractedImages(true);
    }
  };
  console.log(images);
  const handleClassifyImages = async () => {
    const response = await ExtractedService.classify(
      images,
      selectedModel,
      threshold
    );
    navigate("/classify", {
      state: { classificationResults: response.results }, // Truyền kết quả qua state
    });
    // console.log(selectedModel, images)
  };

  const handleReloadPdf = () => {
    setPdfUrl("");
    setIsExtractedImages(false);
    setPdfFile(null);
    if (filePdfRef.current) {
      filePdfRef.current.value = "";
    }
  };

  const handleFindSimilarityImages = () => {};
  return (
    <>
      <p className="mt-2 text-gray-600 text-lg text-center">
        Upload a PDF to extract images effortlessly for further classification
        and similarity search. This feature simplifies the image preparation
        process, enabling efficient analysis and comparison of visually similar
        images
      </p>
      <div className="my-10  p-10 shadow-[0_0_30px_10px_rgba(0,0,255,0.2)] border-blue-400 border-2 border-dashed">
        <div
          className={`grid ${
            isExtractedImages ? "grid-cols-[1fr_4fr]" : "grid-col-1s"
          } `}
        >
          <div className="h-full flex flex-col justify-center items-center ">
            <div className="border border-dashed p-2 h-64 w-52 bg-white mb-5 flex justify-center relative">
              {pdfUrl ? (
                <>
                  <iframe
                    src={pdfUrl}
                    className="w-full h-full"
                    title="PDF preview"
                  ></iframe>
                  <a
                    href={pdfUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="absolute bottom-0 right-6 text-xl text-gray-400"
                  >
                    <IoIosExpand />
                  </a>
                </>
              ) : (
                <p className="text-center">PDF</p>
              )}
            </div>
            <input
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={handleFileChange}
              ref={filePdfRef}
            />
            <div className="">
              {pdfUrl && isExtractedImages == false && (
                <button
                  onClick={handleExtractImages}
                  className="bg-blue-500 rounded-2xl p-2 text-white text-lg 
                                font-bold cursor-pointer"
                >
                  Extract Images
                </button>
              )}
            </div>
            <div>
              {!pdfUrl && isExtractedImages == false && (
                <>
                  <button
                    onClick={() => filePdfRef.current.click()}
                    className="bg-blue-500 rounded-2xl p-2 text-white text-lg font-bold cursor-pointer"
                  >
                    Choose Pdf file
                  </button>
                  {/* <h1 className='text-center'>or drop file here</h1> */}
                </>
              )}
            </div>
          </div>

          {isExtractedImages && images.length > 0 && (
            <>
              <div className="flex flex-col justify-between relative ">
                {pdfUrl && isExtractedImages && (
                  <div
                    className="absolute -top-7 -right-7 z-100"
                    onClick={handleReloadPdf}
                  >
                    <IoReload
                      className="text-blue-600 text-xl cursor-pointer hover:shadow-[0_0_20px_4px_rgba(56,189,248,0.4)] transition-all
                                        duration-300 ease-in-out transform  hover:scale-110 rounded-full"
                    />
                  </div>
                )}
                <div className="grid grid-cols-4 gap-4 mt-5 overflow-y-auto min-h-[200px] max-h-[450px]">
                  {images?.map((image, index) => (
                    <div className="flex flex-col">
                      <img
                        key={index}
                        src={`data:image/png;base64,${image.base64}`}
                        alt={`Extracted image ${index}`}
                        className="w-full h-40 rounded"
                      />
                      <p>
                        <span className="font-bold">Image name:</span>{" "}
                        {image.name}
                      </p>
                      {/* <p>
                        <span className="font-bold">Caption:</span>{" "}
                        {image.caption}
                      </p> */}
                    </div>
                  ))}
                </div>
                <div className="mt-7">
                  <h1 className="font-bold text-blue-600 text-xl mb-4 text-center">
                    Choose Model and Similarity Threshold for Classification and
                    Image Similarity Search
                  </h1>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Model Selector */}
                    <div className="flex flex-col">
                      <label
                        htmlFor="modelSelector"
                        className="text-gray-700 font-semibold mb-2"
                      >
                        Select model:
                      </label>
                      <select
                        id="modelSelector"
                        onChange={(e) => setSelectedModel(e.target.value)}
                        value={selectedModel}
                        className="bg-white border border-gray-300 rounded-xl p-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                      >
                        <option value="vgg16">VGG16</option>
                        <option value="vgg16_aug">VGG16 Augmented</option>
                        <option value="convnext_v2">ConvNeXt V2</option>
                        <option value="convnext_v2_aug">
                          ConvNeXt V2 Augmented
                        </option>
                        <option value="alexnet">AlexNet</option>
                        <option value="alexnet_aug">AlexNet Augmented</option>
                      </select>
                    </div>

                    {/* Threshold Selector */}
                    <div className="flex flex-col">
                      <label
                        htmlFor="similarity"
                        className="text-gray-700 font-semibold mb-2"
                      >
                        Select similarity threshold:
                      </label>
                      <select
                        id="similarity"
                        name="similarity"
                        onChange={(e) => setThreshold(e.target.value)}
                        value={threshold}
                        className="bg-white border border-gray-300 rounded-xl p-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
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

                  {/* Search Button */}
                  <div className="flex justify-center mt-6">
                    <Link to="/classify">
                      <button
                        onClick={handleClassifyImages} // Make sure you implement this function
                        className="bg-blue-500 text-white font-semibold py-3 px-8 rounded-xl shadow-md hover:bg-blue-600 transition duration-300 ease-in-out"
                      >
                        Find Similar Images
                      </button>
                    </Link>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default ExtractedImages;
