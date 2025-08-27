import React, { useMemo } from "react";

function TopPlagiarizedDocs({ filteredImages, onClose, type }) {
  const topDocs = useMemo(
    () => getTopPlagiarizedDocs(filteredImages, type),
    [filteredImages]
  );

  return (
    <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-4   ">
      <div
        className="relative bg-white rounded-2xl shadow-2xl p-6 w-full max-w-7xl    overflow-hidden 
               max-h-[calc(100vh-4rem)] my-10 border border-blue-100"
      >
        <button
          onClick={onClose}
          className="absolute top-5 cursor-pointer  right-3 w-9 h-9 bg-white rounded-full border border-gray-300 hover:bg-red-500 hover:text-white shadow flex items-center justify-center transition-all"
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

        <h2 className="text-2xl font-bold text-rose-600 mt-3 mb-6">
          Top Documents with Most Similar Images
        </h2>
        <div className="h-full max-h-[80vh]   overflow-y-auto    ">
          <div className="h-full">
            <ul className="space-y-6">
              {topDocs.map((doc, idx) => (
                <li
                  key={idx}
                  className="bg-gray-50 border border-gray-200 rounded-lg p-4 shadow-sm"
                >
                  <div className="mb-2">
                    <p className="font-semibold text-lg text-gray-800">
                      {doc.title || "No title"}
                    </p>
                    {doc.doi && (
                      <a
                        href={`https://doi.org/${doc.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 text-sm hover:underline break-words"
                      >
                        {doc.doi}
                      </a>
                    )}
                    <p className="text-sm text-gray-600">
                      {doc.images.length} similar images found
                    </p>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                    {(() => {
                      const seen = new Set();
                      return doc.images
                        .filter((img) => {
                          const key = `${doc.predicted_class}__${img.image_field_name}`;
                          if (seen.has(key)) return false;
                          seen.add(key);
                          return true;
                        })
                        .map((img, i) => (
                          <div
                            key={i}
                            className="border rounded-lg overflow-hidden bg-white shadow-sm p-2"
                          >
                            <img
                              src={`http://127.0.0.1:8000/media/dataset/${img.predicted_class}/${img.image_field_name}`}
                              alt={img.title || ""}
                              className="w-full h-36 object-contain bg-gray-100"
                            />
                          </div>
                        ));
                    })()}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
function getTopPlagiarizedDocs(filteredImages, type) {
  const doiMap = new Map();
  const seenImages = new Set();
  if (type == "image") {
    filteredImages.forEach((info) => {
      const predictedClass = info.predicted_class;

      if (info.similar_images && info.similar_images.length > 0) {
        info.similar_images.forEach((sim) => {
          const doi = sim.doi?.trim();

          if (!doi || doi.toLowerCase().includes("no")) return;

          const imageKey = sim.image_field_name;

          if (seenImages.has(imageKey)) return;

          seenImages.add(imageKey);

          if (!doiMap.has(doi)) {
            doiMap.set(doi, {
              doi,
              title: sim.title,
              images: [],
            });
          }

          doiMap.get(doi).images.push({
            ...sim,
            predicted_class: predictedClass,
          });
        });
      }
    });
  } else {
    const predictedClass = filteredImages.predicted_class;

    const seenImages = new Set();

    filteredImages.similar_images
      .filter((sim) => {
        const isValidDOI =
          sim.doi &&
          sim.doi.trim() !== "" &&
          !sim.doi.toLowerCase().includes("no");

        const imageKey = sim.image_field_name;
        const isUniqueImage = imageKey && !seenImages.has(imageKey);

        if (isValidDOI && isUniqueImage) {
          seenImages.add(imageKey);
          return true;
        }

        return false;
      })
      .forEach((sim) => {
        const doi = sim.doi.trim();
        if (!doiMap.has(doi)) {
          doiMap.set(doi, {
            doi,
            title: sim.title,
            images: [],
          });
        }
        doiMap.get(doi).images.push({
          ...sim,
          predicted_class: predictedClass,
        });
      });
  }

  return Array.from(doiMap.values())
    .sort((a, b) => b.images.length - a.images.length)
    .slice(0, 20);
}

export default TopPlagiarizedDocs;
