import React from "react";

const ReadBeforeUse = () => {
  const listStyle = {
    paddingLeft: "2rem", 
  };
    
  return (
    <div style={{ paddingTop: "2rem" }}>
      <strong>Welcome to the OncoNPC Cancer Type Prediction Tool</strong>
      <p>
        This tool is designed to help researchers and clinicians explore
        potential cancer types based on tumor somatic genetics information,
        using a machine learning algorithm.
      </p>
      <p>
        Please review the following important disclosures before you proceed
        with using this tool:
      </p>
      <div style={{display: 'inline-block', textAlign: 'left'}}>
        <ul style={listStyle}>
          <li>
            This OncoNPC cancer type prediction tool is based on a published
            article in{" "}
            <a href="https://www.nature.com/articles/s41591-023-02482-6">
              Nature Medicine
            </a>
            .
          </li>
          <li>
            It is currently in the experimental stage and has not been validated
            through randomized clinical trials or prospective studies.
          </li>
          <li>
            The tool is intended for research purposes only and should not be
            used as a basis for clinical diagnosis or treatment decisions.
          </li>
          <li>
            It has not been approved for clinical use by any regulatory bodies.
          </li>
          <li>
            No personal data is collected by the tool, and the prediction
            results are processed in a Hugging Face environment based on user
            inputs.
          </li>
        </ul>
      </div>
      <p>
        Please see{" "}
        <a href="https://itmoon7.github.io/onconpc/#/tutorial">this tutorial</a>{" "}
        on how to use this tool.
      </p>
      <p>
        For detailed information regarding the base model, performance metrics,
        implications of the predictions, utilized features, and processing,
        please refer to{" "}
        <a href="https://www.nature.com/articles/s41591-023-02482-6">
          our paper
        </a>
        .
      </p>
      <p>
        For feedback, questions, or to report any issues with the tool, please
        contact <a href="mailto:itmoon@mit.edu">itmoon@mit.edu</a>.
      </p>
      <p>
        Developed by Jennifer Zhou (
        <a href="mailto:jenz33@mit.edu">jenz33@mit.edu</a>) and Intae Moon (
        <a href="mailto:itmoon@mit.edu">itmoon@mit.edu</a>).
      </p>
    </div>
  );
};

export default ReadBeforeUse;
