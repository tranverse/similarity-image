import { Route, Routes, BrowserRouter } from "react-router-dom";
import Home from "@pages/Home";
import './tailwind.css'
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Home/>} path="/"/>
      </Routes>
    </BrowserRouter>
  );
}
export default App