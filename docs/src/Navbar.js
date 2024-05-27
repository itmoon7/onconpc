import React from "react";
import { Link } from "react-router-dom";
import logo from "./onconpc_logo_clipped.png";

const Navbar = () => {
  return (
    <nav
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center", 
        padding: "1rem",
        background: "#bde3f6",
        color: "black",
      }}
    >
      <div style={{ display: "flex", alignItems: "center" }}>
        <img
          src={logo}
          alt="OncoNPC Logo"
          style={{ height: "40px", marginRight: "1rem" }}
        />
        <span>OncoNPC</span>
      </div>
      <div>
        <Link
          to="/readbeforeuse"
          style={{
            color: "black",
            marginRight: "1rem",
            textDecoration: "none",
          }}
        >
          Read Before Use
        </Link>
        <Link
          to="/tutorial"
          style={{
            color: "black",
            marginRight: "1rem",
            textDecoration: "none",
          }}
        >
          Tutorial
        </Link>
        <Link
          to="/prediction"
          style={{
            color: "black",
            marginRight: "1rem",
            textDecoration: "none",
          }}
        >
          Prediction Tool
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
