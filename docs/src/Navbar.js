import React from "react";
import { Link } from "react-router-dom"; // Import Link

const Navbar = () => {
  return (
    <nav
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "1rem",
        background: "#bde3f6",
        color: "black",
      }}
    >
      <div>
        <span>OncoNPC</span>
      </div>
      <div>
        <Link
          to="/tutorial"
          style={{
            color: "black",
            marginRight: "1rem",
            textDecoration: "none",
          }} // Added textDecoration: "none" to mimic <a> tag styling
        >
          Tutorial
        </Link>
        <Link
          to="/prediction"
          style={{
            color: "black",
            marginRight: "1rem",
            textDecoration: "none",
          }} // Added textDecoration: "none" for consistency
        >
          Prediction Tool
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
