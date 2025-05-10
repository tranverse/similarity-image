import React from "react";
import Heading from "./Heading";

const Classification = ({ classificationResults }) => {
  return (
    <>
      <Heading message="Explore Your Image Classification Results" />

      <div className="p-6 my-10 rounded-xl shadow-inner h-[450px] overflow-y-auto space-y-4">
        {classificationResults?.map((result, index) => (
          <div
            key={index}
            className="grid grid-cols-[1fr_2fr] bg-white rounded-xl shadow-md p-4 gap-4"
          >
            <div className="flex items-center justify-center">
              <img
                width={350}
                height={350}
                src={`data:image/jpeg;base64,${result.image}`}
                alt={`Classified ${index}`}
                className="rounded"
              />
            </div>

            <div className="text-gray-800 text-sm space-y-2">
              <p className="text-base font-semibold text-blue-600">
                <strong>Predicted Class:</strong> {result.predicted_class} –{" "}
                <span className="text-blue-600 font-medium">
                  Confidence: {parseFloat(result.confidence).toFixed(2)}%
                </span>
              </p>

              <div className="my-2">
                <p className="font-bold mb-2">Other Class Probabilities:</p>
                <div className="grid grid-cols-2 gap-2">
                  {result?.all_classes
                    ?.filter((cls) => cls.label !== result.predicted_class)
                    ?.sort((a, b) => b.confidence - a.confidence)
                    ?.map((cls, idx) => (
                      <div key={idx} className="space-y-1">
                        <p className="text-gray-600">
                          <strong>{cls.label}</strong>: {cls?.confidence}
                        </p>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default Classification;
