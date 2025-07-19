import React from "react";

const PopupDetails = ({ originalImage, similarImage, metadata, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center ">
      <div
        className="absolute inset-0 bg-black/20 backdrop-blur-none transition-opacity"
        onClick={onClose}
      />

      <div className="relative z-50 bg-white rounded-2xl shadow-xl p-2  md:p-10 w-full max-w-7xl   transition-all">
        <button
          onClick={onClose}
          className="absolute cursor-pointer top-2 right-2 w-6 h-6   shadow hover:bg-red-500 hover:text-white transition-all flex items-center justify-center"
          aria-label="Close"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* ORIGINAL IMAGE */}
          <div className="flex flex-col items-center text-sm text-neutral-700">
            <img
              src={`data:image/jpeg;base64,${originalImage.image}`}
              alt={originalImage.name}
              className="rounded-xl border border-gray-200 shadow-sm h-[320px] w-full object-contain"
            />
            <div className="mt-4 w-full space-y-1">
              <h3 className="text-lg font-semibold text-blue-500 mb-1 text-center">
                Original Image
              </h3>
              <div className="flex gap-2">
                <Info
                  label="Predicted Class"
                  value={originalImage?.predicted_class}
                />{" "}
                {" - "}
                <Info label="Confidence" value={originalImage?.confidence} />
              </div>
              <Info label="Title" value={metadata?.title} />
              <Info label="Authors" value={metadata?.authors} />
              <Info label="Accepted Date" value={metadata?.approved_date} />
              {metadata?.doi && (
                <Info
                  label="DOI"
                  value={
                    <a
                      href={`https://doi.org/${metadata.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline break-all"
                    >
                      {metadata.doi}
                    </a>
                  }
                />
              )}
            </div>
          </div>

          <div className="flex flex-col items-center text-sm text-neutral-700">
            <img
              src={`http://127.0.0.1:8000/media/dataset/${originalImage.predicted_class}/${similarImage.image_field_name}`}
              alt={similarImage.name}
              className="rounded-xl border border-gray-200 shadow-sm h-[320px] w-full object-contain"
            />
            <div className="mt-4 w-full space-y-1 overflow-y-auto h-[250px]  ">
              <h3 className="text-lg font-semibold text-blue-500 mb-1 text-center">
                Similar Image
              </h3>
              <Info
                label="Similarity"
                value={`${(similarImage.similarity * 100).toFixed(2)}%`}
              />
              <Info label="Title" value={similarImage?.title} />
              <Info label="Authors" value={similarImage?.authors} />
              <Info label="Page Number" value={similarImage?.page_number} />
              <Info label="Caption" value={similarImage?.caption} />
              <Info label="Accepted Date" value={similarImage?.approved_date} />
              {similarImage?.doi && (
                <Info
                  label="DOI"
                  value={
                    <a
                      href={`https://doi.org/${similarImage.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline break-all"
                    >
                      {similarImage.doi}
                    </a>
                  }
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const Info = ({ label, value }) => (
  <p className="leading-relaxed break-words">
    <span className="font-medium text-neutral-600">{label}:</span>{" "}
    {value || <span className="text-neutral-400">N/A</span>}
  </p>
);

export default PopupDetails;
