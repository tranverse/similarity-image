import React, { useState } from "react";
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

const ClassifyImages = () => {
  const location = useLocation();

  const { classificationResults } = location.state || {}; // lấy dữ liệu từ state
  if (!classificationResults || classificationResults.length === 0) {
    return (
      <Flex
        vertical
        align="center"
        justify="center"
        style={{ minHeight: "100vh", backgroundColor: "#fff" }}
      >
        <Spin
          size="large"
          tip={
            <span style={{ fontSize: "16px", color: "#555" }}>
              The system is processing to classify and find similar images...
            </span>
          }
        />
      </Flex>
    );
  }

  // Kiểm tra loại model
  const isAugmented =
    classificationResults[0]?.model_type?.includes("aug") &&
    classificationResults[0]?.model_type !== "alexnet_aug";

  return (
    <>
      <Header></Header>
      <Breadcrumb
        items={[
          { label: "Extract Images", to: "/", icon: FaRegFilePdf },
          {
            label: isAugmented
              ? "Classify and Find Similar Images"
              : "Classify",
            to: "/classify",
            icon: TbChartBubble,
          },
        ]}
      ></Breadcrumb>
      <div className="mx-20">
        {isAugmented ? (
          <ImageDetails similarity={classificationResults} />
        ) : (
          <Classification classificationResults={classificationResults} />
        )}
      </div>
    </>
  );
};

export default ClassifyImages;
