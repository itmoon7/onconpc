# OncoNPC
We developed <b>OncoNPC</b> (<b>Onco</b>logy <b>N</b>GS-based <b>P</b>rimary cancer type <b>C</b>lassifier), a molecular cancer type classifier trained on multicenter targeted panel sequencing data. OncoNPC utilized somatic alterations including mutations (single nucleotide variants and indels), mutational signatures, copy number alterations, as well as patient age at the time of sequencing and sex to jointly predict cancer type.

We utilized<br>
<ul>
  <li>R (v4.0.2) and Python (v3.9.13) programming languages</li>
  <li>OncoNPC somatic mutation processing (R deconstructSigs v1.8.0)</li>
  <li>OncoNPC model development and interpretation (Python xgboost v1.2.0, shap v0.41.0)</li>
  <li>Survival analysis (R survival v3.2.7, stats v4.0.2, Python lifelines v0.27.4, scipy v1.7.1)</li>
</ul>

See this <a href="https://github.com/itmoon7/onconpc/blob/main/onconpc_prediction_and_explanation_for_cup_tumors.ipynb"><em>notebook example</em></a> and our <a href="https://www.nature.com/articles/s41591-023-02482-6"><em>manuscript</em></a>.

Citation:
@article{moon2023machine,
  title={Machine learning for genetics-based classification and treatment response prediction in cancer of unknown primary},
  author={Moon, Intae and LoPiccolo, Jaclyn and Baca, Sylvan C and Sholl, Lynette M and Kehl, Kenneth L and Hassett, Michael J and Liu, David and Schrag, Deborah and Gusev, Alexander},
  journal={Nature Medicine},
  pages={1--11},
  year={2023},
  publisher={Nature Publishing Group US New York}
}
