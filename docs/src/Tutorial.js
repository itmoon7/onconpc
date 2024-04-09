import React from "react";
import YouTube from "react-youtube";
import "./TutorialVideo.css";

const Tutorial = () => {
  return (
    <div style={{ paddingTop: "2rem" }}>
      {" "}
      <iframe
        width="1040"
        height="590"
        src="https://www.youtube.com/embed/wYl8W8SrppU"
        title="YouTube video player"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
      ></iframe>
    </div>
  );
};

export default Tutorial;
