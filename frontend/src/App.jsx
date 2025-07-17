import { Route, Routes, BrowserRouter } from "react-router-dom";
import Home from "@pages/Home";
import "./tailwind.css";
import ClassifyImages from "@pages/ClassifyImages";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import SingleImageSimilarPage from "@pages/SingleImageSimilarPage";
function App() {
  return (
    <BrowserRouter>
      <ToastContainer position="top-right" autoClose={3000} />

      <Routes>
        <Route element={<Home />} path="/" />
        <Route element={<ClassifyImages />} path="/classify" />
        <Route element={<SingleImageSimilarPage />} path="/single-image" />
      </Routes>
    </BrowserRouter>
  );
}
export default App;
