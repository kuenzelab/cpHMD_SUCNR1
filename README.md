# SUCNR1 - cpHMD of SUCNR1 to investigate selective G protein signaling

This repository contains all scripts and exemplary data regarding the study "...". It uses the cpHMD incorporation in Gromacs from Noora Aho et al. (https://gitlab.com/gromacs-constantph/constantph).

## Files in Analysis

* 01_extract_lambda.sh -> Extracts the lambda coordinates of each titratable site in the simulation
* 02_classify_frames_trajectory_new.py -> sorts the trajectory files to different groups depending on the titration states in the binding site (defined in config.yaml), for all other titration sites the protonation state is set to the most visited in the whole trajectory; generates a topology for all groups
* contact_fraction_neu_sidechain.py -> Calculates the contact fraction of defined residues with the user selected ligand
* interaction_analysis_neu.py -> Calculates the contact fraction to all other residues the defined residue has

These scripts were written with the help of Claude (Sonnet 5.5, Anthropic).

## Exemplary data
