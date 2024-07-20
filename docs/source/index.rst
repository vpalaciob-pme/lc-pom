LCPOM
******

Polarized Optical Microscopy (POM) is crucial for studying materials with internal microstructure, as it reveals supermolecular textures and structures produced by collective ordering. **LCPOM** is a Python software package developed to bridge the gap between experimental observations and theoretical simulations by accurately reproducing color POM images using analytical or simulated order fields, specific material properties, and realistic lab-specific setups. Despite advances in imaging techniques, achieving precise and customizable visualizations of anisotropic materials remains challenging. Traditional methods often lack the flexibility and accuracy needed to adapt to diverse experimental conditions and material characteristics. The challenges are twofold: i) modeling the diverse optical behaviors of different materials accurately and ii) ensuring the simulations run efficiently on modern computing architectures to produce meaningful insights. By bridging the gap between theoretical simulations and practical experimentation, **LCPOM** significantly enhances our ability to study, visualize, and optimize the optical properties of anisotropic materials.

Key Features
============
**LCPOM** is capable of reproducing color images of anisotropic materials by involving:
     - Material-specific optical properties
     - Different incident light options: uniform, LED lamp, Iridiscent, and user-defined
     - Transmittance modes to account for loss of signal when light crosses a curved boundary
     - Human perception of color is taken into account 

Use Cases
=========
The development of **LCPOM** is motivated by the work displayed by liquid crystals under confinement. However, the principles for calculating color POM can be extended to other systems with anisotropic optical properties. The `de Pablo Group <https://pme.uchicago.edu/group/de-pablo-group>`_ has successfully used **LCPOM** to reproduce experimental images of Liquid Crystalline Elastomers, Nematic LCs, Cholesteric LCs, and Skyrmions.

We invite you to explore the gallery and publications for a complete list of others using the package.

.. attention::
     This project is under active development.

==================
Documentation
==================

.. toctree::
     :titlesonly:
     :caption: Contents
     
     userguide
     advuse
     examples
     faq
     troubleshoot
     publications
     about
     api
     glossary

