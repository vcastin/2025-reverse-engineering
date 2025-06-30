# 2025-reverse-engineering
Codes de l'article "Ouvrons la boîte noire : reconstruction de réseaux de neurones parcimonieux par échantillonnage". 

`shallow_reconstruction.py` implémente la reconstruction des paramètres d'un MTP à une couche cachée, décrite en section 4 de l'article, et produit la Figure 2.

`butterfly_reconstruction.py` implémente la reconstruction des paramètres d'un MTP butterfly multicouche, et calcule l'erreur relative médiane commise par le réseau reconstruit sur un grand nombre d'échantillons aléatoires.

___________________________________

Codes for the paper "Opening the Black Box: Reverse-Engineering of Sparse Neural Networks".

`shallow_reconstruction.py` implements the reverse-engineering of a two-layer MTP, as described in Section 4 of the paper, and produces Figure 2.

`butterfly_reconstruction.py` implements the reverse-engineering of a multilayer butterfly network, and computes the median relative error of the reconstructed network over a large number of random samples.
