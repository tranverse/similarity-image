import React from "react";

const PopupDetails = ({ originalImage, similarImage, metadata, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay nền tối với blur */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Hộp nội dung popup */}
      <div className="relative z-50 bg-white rounded-2xl shadow-xl p-6 md:p-10 w-full max-w-6xl mx-4 transition-all">
        {/* Nút đóng */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-9 h-9 rounded-full bg-white border border-gray-200 hover:bg-neutral-100 text-gray-500 hover:text-red-500 transition flex items-center justify-center text-xl"
          aria-label="Close"
        >
          &times;
        </button>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* ORIGINAL IMAGE */}
          <div className="flex flex-col items-center text-sm text-neutral-700">
            <img
              src={`data:image/jpeg;base64,${originalImage.image}`}
              alt={originalImage.name}
              className="rounded-xl border border-gray-200 shadow-sm max-h-[320px] object-contain"
            />
            <div className="mt-4 w-full space-y-1">
              <h3 className="text-lg font-medium text-neutral-800 mb-2">🎯 Original Image</h3>
              <Info label="Predicted Class" value={originalImage?.predicted_class} />
              <Info label="Confidence" value={originalImage?.confidence} />
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
                      className="text-blue-600 hover:underline"
                    >
                      {metadata.doi}
                    </a>
                  }
                />
              )}
            </div>
          </div>

          {/* SIMILAR IMAGE */}
          <div className="flex flex-col items-center text-sm text-neutral-700">
            <img
              src={`http://127.0.0.1:8000/media/dataset/${originalImage.predicted_class}/${similarImage.image_field_name}`}
              alt={similarImage.name}
              className="rounded-xl border border-gray-200 shadow-sm max-h-[320px] object-contain"
            />
            <div className="mt-4 w-full space-y-1">
              <h3 className="text-lg font-medium text-neutral-800 mb-2">🧩 Similar Image</h3>
              <Info label="Similarity" value={`${(similarImage.similarity * 100).toFixed(2)}%`} />
              <Info label="Title" value={similarImage?.title} />
              <Info label="Authors" value={similarImage?.authors} />
              <Info label="Page Number" value={similarImage?.page_number} />
              <Info label="Caption" value={similarImage?.caption} />
              {similarImage?.doi && (
                <Info
                  label="DOI"
                  value={
                    <a
                      href={`https://doi.org/${similarImage.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline"
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
  <p>
    <span className="font-medium text-neutral-600">{label}:</span>{" "}
    {value || <span className="text-neutral-400">N/A</span>}
  </p>
);

export default PopupDetails;
