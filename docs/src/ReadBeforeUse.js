import React from "react";

const ReadBeforeUse = () => {
  return (
    <div style={{ paddingTop: "2rem" }}>
      **Welcome to the OncoNPC Cancer Type Prediction Tool** This tool is
      designed to help researchers and clinicians explore potential cancer types
      based on tumor somatic genetics information, using a machine learning
      algorithm. Please review the following important disclosures before you
      proceed with using this tool: This OncoNPC cancer type prediction tool is
      based on a published article in [Nature
      Medicine](https://www.nature.com/articles/s41591-023-02482-6). It is
      currently in the experimental stage and has not been validated through
      randomized clinical trials or prospective studies. The tool is intended
      for research purposes only and should not be used as a basis for clinical
      diagnosis or treatment decisions. It has not been approved for clinical
      use by any regulatory bodies. No personal data is collected by the tool,
      and the prediction results are processed in a Hugging Face environment
      based on user inputs. Please see [this
      tutorial](https://itmoon7.github.io/onconpc/#/tutorial) on how to use this
      tool. For detailed information regarding the base model, performance
      metrics, implications of the predictions, utilized features, and
      processing, please refer to [our
      paper](https://www.nature.com/articles/s41591-023-02482-6). For feedback,
      questions, or to report any issues with the tool, please contact
      itmoon@mit.edu. Developed by Jennifer Zhou (jenz33@mit.edu) and Intae Moon
      (itmoon@mit.edu)
    </div>
  );
};

export default ReadBeforeUse;
