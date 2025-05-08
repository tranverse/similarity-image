import { Route, Routes, BrowserRouter } from "react-router-dom";
import Home from "@pages/Home";
import './tailwind.css'
import ClassifyImages from "@pages/ClassifyImages";
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Home/>} path="/"/>
        <Route element={<ClassifyImages/>} path="/classify"/>

      </Routes>
    </BrowserRouter>
  );
}
export default App