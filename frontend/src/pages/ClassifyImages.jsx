import React from "react";
import Classification from "@components/Classification";
import ImageDetails from "@components/ImageDetails";
import Header from "@components/Header";
import Heading from "@components/Heading";
import Breadcrumb from "@components/Breadcrumb";
import { TbChartBubble } from "react-icons/tb";
import { FaRegFilePdf } from "react-icons/fa";

const ClassifyImages = () => {
  return (
    <>
      <Header></Header>
      <Breadcrumb items={[
        {label : 'Extract Images', to: '/', icon: FaRegFilePdf},
        {label: 'Classify', to: '/classify', icon: TbChartBubble}
      ]}></Breadcrumb>

      <div className="mx-20">
        <Heading message="Classify Images"></Heading>
        <Classification></Classification>
        <Heading message="Find Similariy Images"></Heading>
        <ImageDetails></ImageDetails>
      </div>
    </>
  );
};

export default ClassifyImages;
