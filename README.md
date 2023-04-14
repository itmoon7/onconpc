# onconpc
We developed <b>OncoNPC</b> (\textbf{Onco}logy \textbf{N}GS-based \textbf{P}rimary cancer type \textbf{C}lassifier), a molecular cancer type classifier trained on multicenter targeted panel sequencing data (Fig. \ref{fig:cup_method}). OncoNPC utilized somatic alterations including mutations (single nucleotide variants and indels), mutational signatures, copy number alterations, as well as patient age at the time of sequencing and sex to jointly predict cancer type using a XGBoost algorithm

We utilized the R (v4.0.2) and Python (v3.9.13) programming languages for OncoNPC feature processing (R deconstructSigs v1.8.0), OncoNPC model development and interpretation (Python xgboost v1.2.0, shap v0.41.0), and survival analysis (R survival v3.2.7, stats v4.0.2, Python lifelines v0.27.4, scipy v1.7.1)

See <a href="https://www.medrxiv.org/content/10.1101/2022.12.22.22283696v1"><em>here</em></a> for our preprint.

Citation:
@article{moon2022utilizing,
  title={Utilizing Electronic Health Records (EHR) and Tumor Panel Sequencing to Demystify Prognosis of Cancer of Unknown Primary (CUP) patients},
  author={Moon, Intae and LoPiccolo, Jaclyn and Baca, Sylvan C and Sholl, Lynette M and Kehl, Kenneth L and Hassett, Michael J and Liu, David and Schrag, Deborah and Gusev, Alexander},
  journal={medRxiv},
  pages={2022--12},
  year={2022},
  publisher={Cold Spring Harbor Laboratory Press}
}
