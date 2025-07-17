import { Link, useLocation } from "react-router-dom";
import { ChevronRight } from "lucide-react";

const Breadcrumb = ({ items }) => {
  const location = useLocation(); // ✅ Bật lại

  return (
    <nav aria-label="Breadcrumb" className="mt-5 mx-5">
      <ol className="flex flex-wrap items-center text-sm text-blue-600">
        {items.map((item, index) => {
          const isActive = item.to === location.pathname;

          return (
            <li key={index} className="flex items-center space-x-2">
              {index > 0 && (
                <ChevronRight className="w-4 h-4 text-blue-400" />
              )}

              <div
                className={`flex items-center space-x-1 rounded-2xl px-3 py-1 ${
                  isActive ? "bg-blue-500 text-white" : ""
                }`}
              >
                {item.icon && (
                  <item.icon
                    className={`w-4 h-4 ${
                      isActive ? "text-white" : "text-blue-500"
                    }`}
                  />
                )}
                {item.to ? (
                  <Link
                    to={item.to}
                    className={`transition-colors duration-200 font-medium ${
                      isActive
                        ? "text-white cursor-default pointer-events-none"
                        : "hover:  hover:no-underline"
                    }`}
                  >
                    {item.label}
                  </Link>
                ) : (
                  <span
                    className={`font-semibold ${isActive ? "text-white" : ""}`}
                  >
                    {item.label}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

export default Breadcrumb;
