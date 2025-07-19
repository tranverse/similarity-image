import React, { useEffect, useState } from "react";
import Classification from "@components/Classification";
import ImageDetails from "@components/ImageDetails";
import Header from "@components/Header";
import Heading from "@components/Heading";
import Breadcrumb from "@components/Breadcrumb";
import { TbChartBubble } from "react-icons/tb";
import { FaRegFilePdf } from "react-icons/fa";
import { useLocation } from "react-router-dom";
import CircularProgress from "@mui/material/CircularProgress";
import { Flex, Spin } from "antd";
import { toast } from "react-toastify";
import ExtractedService from "@services/Extracted.service";

const ClassifyImages = () => {
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(false);
  const { chosenImages, metadata, selectedModel, threshold } =
    location.state || [];
  const [classificationResults, setClassificationResults] = useState([]);

  useEffect(() => {
    handleClassifyImages();
  }, []);
  const handleClassifyImages = async () => {
    setIsLoading(true);
    try {
      const response = await ExtractedService.classify(
        chosenImages,
        selectedModel,
        threshold
      );
      if (response.error) {
        toast.error("An error occurred during image classification!");
        return;
      }

      if (!response.results || response.results.length === 0) {
        toast.info("No matching results found.");
        return;
      }
      setClassificationResults(response.results);
    } catch (error) {
      toast.error("Error");
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 min-h-screen ">
        <span className=" loading loading-infinity loading-xl  scale-200 bg-gradient-to-r from-blue-300 to-blue-700"></span>
        <p className="font-semibold  text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-blue-700">
          Please wait while the system finds similar images...
        </p>
      </div>
    );
  }

  const isAugmented = classificationResults[0]?.model_type?.includes("aug");

  return (
    <>
      <div className="bg-gradient-to-br from-blue-50 via-white to-rose-50 min-h-screen">
        <Header></Header>
        <Breadcrumb
          items={[
            { label: "Extract Images", to: "/pdf", icon: FaRegFilePdf },
            {
              label: isAugmented
                ? "Classify and Find Similar Images"
                : "Classify",
              to: "/classify",
              icon: TbChartBubble,
            },
          ]}
        ></Breadcrumb>
        <div className="mx-5  ">
          {isAugmented ? (
            <ImageDetails
              similarity={classificationResults}
              metadata={metadata}
            />
          ) : (
            <Classification classificationResults={classificationResults} />
          )}
        </div>
      </div>
    </>
  );
};

export default ClassifyImages;
