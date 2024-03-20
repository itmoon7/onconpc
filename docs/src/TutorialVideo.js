import React from "react";
import YouTube from "react-youtube";

function TutorialVideo() {
  const opts = {
    height: "390",
    width: "640",
    playerVars: {
      autoplay: 1, // Auto-play the video on load
    },
  };

  return <YouTube videoId="n4MbunbrnjQ" opts={opts} />;
}

export default TutorialVideo;
