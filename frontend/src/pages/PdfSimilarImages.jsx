import React from "react";
import Header from "@components/Header";
import Heading from "@components/Heading";
import ExtractedImages from "@components/ExtractedImages";
import Breadcrumb from "@components/Breadcrumb";
import { FaRegFilePdf } from "react-icons/fa";
const PdfSimilarImages = () => {
  return (
    <>
      <div className="bg-gradient-to-br from-blue-50 via-white to-rose-50 min-h-screen">
        <Header></Header>
        {/* <Breadcrumb
          items={[{ label: "Extract Images", to: "/", icon: FaRegFilePdf }]}
        ></Breadcrumb> */}
        <div className=" mx-5 my-10 ">
          <ExtractedImages></ExtractedImages>
        </div>
      </div>
    </>
  );
};

export default PdfSimilarImages;
