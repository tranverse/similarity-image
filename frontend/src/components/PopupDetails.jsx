import React from "react";

const PopupDetails = ({ originalImage, similarImage, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Nền mờ với hiệu ứng blur nhẹ và overlay bán trong suốt */}
      <div
        className="absolute inset-0 bg-white/30 backdrop-blur-sm z-40 transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Nội dung popup */}
      <div className="relative z-50 bg-white rounded-2xl shadow-2xl p-8 max-w-6xl w-full mx-4 grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Nút đóng */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-500 hover:text-red-500 text-3xl font-bold transition"
          aria-label="Close"
        >
          ×
        </button>

        {/* Original Image */}
        <div className="flex flex-col items-center text-center">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Original Image</h2>
          <img
            src={`data:image/jpeg;base64,${originalImage.image}`}
            alt={originalImage.name}
            className="rounded-xl max-h-64 object-contain border border-gray-300 shadow mb-4"
          />
          <div className="text-sm text-gray-700 space-y-1 w-full">
            <p><strong>Name:</strong> {originalImage.name}</p>
            <p><strong>Caption:</strong> {originalImage.caption || "N/A"}</p>
            {originalImage.doi && (
              <a
                href={`https://doi.org/${originalImage.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                DOI: {originalImage.doi}
              </a>
            )}
          </div>
        </div>

        {/* Similar Image */}
        <div className="flex flex-col items-center text-center">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Similar Image</h2>
          <img
            src={`http://127.0.0.1:8000/media/raw/${originalImage.predicted_class}/${similarImage.image_field_name}`}
            alt={similarImage.name}
            className="rounded-xl max-h-64 object-contain border border-gray-300 shadow mb-4"
          />
          <div className="text-sm text-gray-700 space-y-1 w-full">
            <p><strong>Name:</strong> {similarImage.name}</p>
            <p><strong>Caption:</strong> {similarImage.caption || "N/A"}</p>
            <p><strong>Similarity:</strong> {(similarImage.similarity * 100).toFixed(2)}%</p>
            {similarImage.doi && (
              <a
                href={`https://doi.org/${similarImage.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                DOI: {similarImage.doi}
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PopupDetails;
