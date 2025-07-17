import React from "react";
import Header from "@components/Header";
import Heading from "@components/Heading";
import ExtractedImages from "@components/ExtractedImages";
import Breadcrumb from "@components/Breadcrumb";
import { FaRegFilePdf } from "react-icons/fa";
const Home = () => {
  return (
    <>
      <div className="bg-[#f9f6f2] min-h-screen">
        <Header></Header>
        <Breadcrumb
          items={[{ label: "Extract Images", to: "/", icon: FaRegFilePdf }]}
        ></Breadcrumb>
        <div className=" mx-5 ">
          <ExtractedImages></ExtractedImages>
        </div>
      </div>
    </>
  );
};

export default Home;
