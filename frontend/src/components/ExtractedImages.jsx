import React, { useEffect, useRef, useState } from "react";
import Button from "./Button";
import Photograph from "@assets/images/01-5689-A-VAN HUU HUE-1-10_page_2_img_1.png";
import ExtractedService from "@services/Extracted.service";
import { Link, useNavigate } from "react-router-dom";
import { IoReload } from "react-icons/io5";
import { IoIosExpand } from "react-icons/io";
import Heading from "./Heading";
import { FaFilePdf } from "react-icons/fa6";
import { IoIosImages } from "react-icons/io";
import { TbArrowBigRightLines } from "react-icons/tb";
import pdf from "@assets/images/pdf.png";
import imagespng from "@assets/images/image.png";
import to from "@assets/images/next.png";
import { FaRegCheckCircle } from "react-icons/fa";
import { FaCheck } from "react-icons/fa";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { SlTrash } from "react-icons/sl";

const ExtractedImages = () => {
  const [pdfUrl, setPdfUrl] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const filePdfRef = useRef(null);
  const [isExtractedImages, setIsExtractedImages] = useState(false);
  const [imagedata, setImageData] = useState([]);
  const [selectedModel, setSelectedModel] = useState("vgg16_aug");
  const [threshold, setThreshold] = useState(0.7);
  const [chosenImages, setChosenImages] = useState([]);
  const [isSelectAll, setIsSelectAll] = useState(false);
  const [isLoadImages, setIsLoadImages] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const savedState = sessionStorage.getItem("extractedState");
    if (savedState) {
      const {
        pdfUrl,
        selectedModel,
        threshold,
        chosenImages,
        imagedata,
        isExtractedImages,
      } = JSON.parse(savedState);
      setPdfUrl(pdfUrl);
      setSelectedModel(selectedModel);
      setThreshold(threshold);
      setChosenImages(chosenImages);
      setImageData(imagedata);
      setIsExtractedImages(isExtractedImages);
    }
  }, []);
  console.log(chosenImages)
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "application/pdf") {
      const reader = new FileReader();
      reader.onload = () => {
        const base64Pdf = reader.result;
        setPdfUrl(base64Pdf);
        setPdfFile(file);

        const stateToSave = {
          pdfUrl: base64Pdf,
          selectedModel,
          threshold,
          chosenImages: newChosen,
          imagedata,
          isExtractedImages,
        };
        sessionStorage.setItem("extractedState", JSON.stringify(stateToSave));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleExtractImages = async () => {
    setIsLoadImages(true);

    try {
      const data = await ExtractedService.sendPdf({ pdf: pdfFile });

      setImageData(data);
      if (data) {
        setIsExtractedImages(true);
      }
    } catch (error) {
      console.error(error);
      toast.error("Failed to extract images from PDF. Please try again.");
    } finally {
      setIsLoadImages(false);
    }
  };

  const handleClassifyImages = async () => {
    if (chosenImages.length == 0) {
      toast.warning("Please select at least one image");
      return;
    }

    const stateToSave = {
      pdfUrl,
      selectedModel,
      threshold,
      chosenImages,
      imagedata,
      isExtractedImages,
    };
    sessionStorage.setItem("extractedState", JSON.stringify(stateToSave));

    navigate("/classify", {
      state: {
        metadata: imagedata.metadata,
        chosenImages: chosenImages,
        selectedModel: selectedModel,
        threshold: threshold,
      },
    });
  };

  const handleReloadPdf = () => {
    sessionStorage.removeItem("extractedState");

    setPdfUrl("");
    setIsExtractedImages(false);
    setPdfFile(null);
    if (filePdfRef.current) {
      filePdfRef.current.value = "";
    }
  };

  const handleSelectAllImages = () => {
    if (isSelectAll) {
      setChosenImages([]);
    } else {
      setChosenImages([...imagedata?.images]);
    }
    setIsSelectAll(!isSelectAll);
  };

  const handleChooseImage = (image) => {
    let newChosen;

    const isSelected = chosenImages.some((img) => img.index === image.index);
    if (isSelected) {
      newChosen = chosenImages.filter((img) => img.index !== image.index);
    } else {
      newChosen = [...chosenImages, image];
    }

    setChosenImages(newChosen);

    // Cập nhật sessionStorage
    const stateToSave = {
      pdfUrl,
      selectedModel,
      threshold,
      chosenImages: newChosen,
      imagedata,
      isExtractedImages,
    };
    sessionStorage.setItem("extractedState", JSON.stringify(stateToSave));
  };

  return (
    <>
      <div className=" my-2">
        {isExtractedImages == false ? (
          <div className="p-5 rounded-2xl border border-gray-200 shadow-md  bg-white h-[600px]">
            <div className="p-2 border border-dashed border-gray-300 rounded-lg h-full">
              {pdfUrl && !isLoadImages ? (
                <>
                  <div className=" flex justify-center ">
                    <iframe
                      src={pdfUrl}
                      className=" m-2 flex  justify-center items-center border  border-dashed rounded-lg  w-[500px] h-[480px] cursor-pointer"
                    ></iframe>
                    <div className="mt-2">
                      <SlTrash
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReloadPdf();
                        }}
                        className="cursor-pointer text-2xl text-blue-500"
                      />
                    </div>
                  </div>
                </>
              ) : (
                <div
                  className={` flex ${
                    isLoadImages ? "justify-center" : "flex-col justify-between"
                  } items-center h-full`}
                >
                  {isLoadImages == true ? (
                    <div className="flex flex-col items-center justify-center gap-2 ">
                      <span className=" loading loading-infinity loading-xl  scale-200 bg-gradient-to-r from-blue-300 to-blue-700"></span>
                      <p className="font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-blue-700">
                        Wait for the system to extract images
                      </p>
                    </div>
                  ) : (
                    <>
                      <div className="flex flex-col items-center justify-center">
                        <div className="">
                          <img src={imagespng} className="h-[150px]" alt="" />
                        </div>
                        <h1 className="text-4xl text-blue-400 mt-10 font-semibold">
                          Extract images from PDF
                        </h1>
                        <p className="text-center text-gray-600">
                          Extract images and information from scientific
                          articles published in the CTU Journal of Science
                        </p>
                        <input
                          type="file"
                          accept="application/pdf"
                          className="hidden"
                          onChange={handleFileChange}
                          ref={filePdfRef}
                        />
                      </div>
                      <div
                        className="mb-10 flex flex-col items-center"
                        onClick={() => filePdfRef.current.click()}
                      >
                        <button className="bg-blue-400 px-8 py-4 text-white text-xl rounded-lg font-semibold cursor-pointer">
                          Choose a PDF file
                        </button>
                        <p className="text-gray-600 mt-2">
                          or drag and drop a file
                        </p>
                      </div>
                    </>
                  )}
                </div>
              )}
              {pdfUrl && !isLoadImages && (
                <div className="bg-blue-400 rounded-lg p-2 ">
                  <button
                    className="w-full text-white text-lg font-semibold cursor-pointer"
                    onClick={() => {
                      handleExtractImages();
                    }}
                  >
                    Extract images
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex gap-4 rounded-2xl h-[600px]  ">
            <div className="w-1/3 h-full flex flex-col  gap-2  justify-between ">
              <div className="flex-1 p-2 shadow bg-white h-full rounded-lg overflow-y-auto  ">
                <div className=" flex justify-end">
                  <SlTrash
                    onClick={handleReloadPdf}
                    className="cursor-pointer text-2xl text-blue-500"
                  />
                </div>
                <div className="h-2/4 p-2 rounded-lg ">
                  <iframe
                    src={pdfUrl}
                    className=" w-full h-full   cursor-pointer"
                  ></iframe>
                </div>
                <div className="h-2/4   text-sm p-2 rounded-lg">
                  <p className="text-gray-700">
                    <strong>Title: </strong> {imagedata?.metadata?.title}
                  </p>
                  <p className="text-gray-700">
                    <strong>Authors: </strong> {imagedata?.metadata?.authors}
                  </p>
                  <p className="text-gray-700">
                    {" "}
                    <strong>Accepted Date:</strong>{" "}
                    {imagedata?.metadata?.approved_date}
                  </p>
                  <p className="text-gray-700">
                    <strong>DOI: </strong>
                    {imagedata?.metadata?.doi == "NO DOI" ? (
                      <strong className="text-blue-500 hover:text-blue-700">
                        {imagedata?.metadata?.doi}
                      </strong>
                    ) : (
                      <a
                        target="_blank"
                        href={`https://doi.org/${imagedata?.metadata?.doi}`}
                        className="text-blue-500 hover:text-blue-700"
                      >
                        {imagedata?.metadata?.doi}
                      </a>
                    )}
                  </p>
                </div>
              </div>
              <div className="bg-white p-3 shadow rounded-lg">
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

                <div className=" justify-center mt-2 ">
                  <button
                    disabled={imagedata?.images?.length == 0}
                    onClick={handleClassifyImages}
                    className={`bg-blue-500 text-white font-semibold py-2 px-8 rounded-md   w-full 
                      hover:bg-blue-400 transition duration-300 ease-in-out
                      ${
                        imagedata?.images?.length == 0
                          ? "opacity-50 cursor-not-allowed"
                          : "cursor-pointer"
                      }`}
                  >
                    Find Similar Images
                  </button>
                </div>
              </div>
            </div>

            <div className="w-2/3 flex flex-col rounded-lg py-3  shadow bg-white">
              {imagedata?.images?.length === 0 ? (
                <div className="flex flex-col flex-1 items-center justify-center py-12 text-center  text-yellow-800 rounded-xl ">
                  <p className="text-lg font-medium">
                    No images found in the PDF.
                  </p>
                  <p className="text-sm text-gray-600">
                    Please choose another file.
                  </p>
                </div>
              ) : (
                <>
                  <div className="flex justify-between items-center px-3">
                    <p className="text-blue-500 font-semibold">
                      Extrated images: {imagedata?.images?.length}
                    </p>
                    <button
                      onClick={handleSelectAllImages}
                      className={`flex gap-1 items-center   font-semibold px-2 py-1 rounded-lg  border focus:outline-none
                      cursor-pointer border-blue-500 hover:text-white hover:bg-blue-400
                   ${
                     isSelectAll
                       ? "bg-blue-500 text-white "
                       : "bg-white text-blue-500"
                   } `}
                    >
                      <FaRegCheckCircle className="text-[15px]  " />
                      Select All
                    </button>
                  </div>
                  <div className="p-3 grid grid-cols-2 md:grid-cols-4 gap-4 cursor-pointer overflow-x-hidden  overflow-y-auto mt-2 ">
                    {imagedata?.images?.map((img, index) => (
                      <div
                        onClick={() => handleChooseImage(img)}
                        className="border bg-white border-gray-300 rounded-lg hover:shadow-lg p-2 relative overflow-visible
                     group duration-300 **: "
                        key={index}
                      >
                        <div
                          className={` rounded-full shadow   w-6 h-6 absolute -top-2 group-hover:opacity-100 opacity-0 
                    -right-2 z-20 flex items-center justify-center
                     ${
                       chosenImages.some((chosen) => chosen.index === img.index)
                         ? "bg-green-300 opacity-100 "
                         : "bg-gray-100 border border-gray-200"
                     }`} 
                        >
                          {chosenImages.some((chosen) => chosen.index === img.index) && (
                            <FaCheck className="text-white" />
                          )}
                        </div>

                        <div className="aspect-square ">
                          <img
                            src={`data:image/png;base64,${img?.base64}`}
                            className="w-full h-full object-contain "
                            alt=""
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default ExtractedImages;
