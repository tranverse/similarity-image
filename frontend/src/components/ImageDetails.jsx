import React from "react";
import Photograph from "@assets/images/01-5689-A-VAN HUU HUE-1-10_page_2_img_1.png";
const ImageDetails = () => {
  return (
    <>
      <div className="flex my-10">


        <div className="bg-red-100 p-4 ">
          <div className="">
            <img src={Photograph} alt="" width={350} height={350} className="rounded"/>
          </div>

            <div>
                <h1 className="text-center">Information</h1>
                <p>Name: </p>
                <a href="">Doi: </a>
                <p>Caption:</p>
            </div>

        </div>


        <div className="bg-green-200">

            <div className="flex ">
                <div className="m-3">
                    <img src={Photograph} alt="" width={250} height={200} className="rounded"/>
                    <p>Score: </p>
                    <h1 className="text-center">Information</h1>
                    <p>Name: </p>
                    <a href="">Doi: </a>
                    <p>Caption:</p>
                </div>
                <div className="m-3">
                    <img src={Photograph} alt="" width={250} height={200} className="rounded"/>
                    <p>Score: </p>
                    <h1 className="text-center">Information</h1>
                    <p>Name: </p>
                    <a href="">Doi: </a>
                    <p>Caption:</p>
                </div>
                <div className="m-3">
                    <img src={Photograph} alt="" width={250} height={200} className="rounded"/>
                    <p>Score: </p>
                    <h1 className="text-center">Information</h1>
                    <p>Name: </p>
                    <a href="">Doi: </a>
                    <p>Caption:</p>
                </div>
                <div className="m-3">
                    <img src={Photograph} alt="" width={250} height={200} className="rounded"/>
                    <p>Score: </p>
                    <h1 className="text-center">Information</h1>
                    <p>Name: </p>
                    <a href="">Doi: </a>
                    <p>Caption:</p>
                </div>
            </div>
        </div>
      </div>
    </>
  );
};

export default ImageDetails;
