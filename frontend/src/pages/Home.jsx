import React from "react";
import Header from "@components/Header";
import Heading from "@components/Heading";
import ExtractedImages from "@components/ExtractedImages";
import Breadcrumb from "@components/Breadcrumb";
import { FaRegFilePdf } from "react-icons/fa";
const Home = () => {
  return (
    <>
      <Header></Header>
      <Breadcrumb items={[{label : 'Extract Images', to: '/', icon: FaRegFilePdf}]}></Breadcrumb>
      <div className="min-h-screen mx-20">
        <div>
          <Heading message="Extract PDF images"></Heading>
          
          <ExtractedImages></ExtractedImages>
        </div>
      </div>
    </>
  );
};

export default Home;
