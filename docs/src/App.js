import logo from "./logo.svg";
import "./App.css";
import Navbar from "./Navbar";
import Gradio from "./Gradio";
import Tutorial from "./Tutorial";
import { HashRouter as Router, Route, Routes } from "react-router-dom";

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<Gradio />} />
          <Route path="/prediction" element={<Gradio />} />
          <Route path="/tutorial" element={<Tutorial />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
