import React from "react";

const PopupDetails = ({ originalImage, similarImage, metadata, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      <div className="relative z-50 bg-white rounded-3xl shadow-2xl p-4 md:p-10 w-full max-w-7xl transition-all border border-blue-100">
        <button
          onClick={onClose}
          className="absolute top-3 right-3 w-9 h-9 bg-white rounded-full border border-gray-300 hover:bg-red-500 hover:text-white shadow flex items-center justify-center transition-all"
          aria-label="Close"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        <h2 className="text-2xl font-bold text-center text-indigo-600 mb-6">
          📸 Image Comparison Details
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* ORIGINAL IMAGE */}
          <div className="flex flex-col items-center text-sm text-neutral-700">
            <img
              src={`data:image/jpeg;base64,${originalImage.image}`}
              alt={originalImage.name}
              className="rounded-xl border border-gray-200 shadow-md h-[320px] w-full object-contain bg-gray-50"
            />
            <div className="mt-4 w-full space-y-2">
              <h3 className="text-lg font-semibold text-indigo-500 mb-1 text-center">
                Original Image
              </h3>
              <div className="flex gap-2 justify-center text-sm">
                <p>
                  <span className="text-gray-600 font-medium">
                    Predicted Class:
                  </span>{" "}
                  <span className="text-emerald-500 font-semibold">
                    {originalImage?.predicted_class || "N/A"}
                  </span>
                </p>
                <span className="text-gray-400">–</span>
                <p>
                  <span className="text-gray-600 font-medium">Confidence:</span>{" "}
                  <span className="text-emerald-500 font-semibold">
                    {originalImage?.confidence || "N/A"}
                  </span>
                </p>
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
                      className="text-indigo-600 hover:underline break-all"
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
              className="rounded-xl border border-gray-200 shadow-md h-[320px] w-full object-contain bg-gray-50"
            />
            <div className="mt-4 w-full space-y-2 overflow-y-auto max-h-[250px]">
              <h3 className="text-lg font-semibold text-indigo-500 mb-1 text-center">
                Similar Image
              </h3>
              <div className="flex justify-center">
                <p className="text-base font-bold text-emerald-500">
                  Similarity: {(similarImage.similarity * 100).toFixed(2)}%
                </p>
              </div>

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
                      className="text-indigo-600 hover:underline break-all"
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
    <span className="font-medium text-neutral-700">{label}:</span>{" "}
    {value || <span className="text-neutral-400">N/A</span>}
  </p>
);

export default PopupDetails;
