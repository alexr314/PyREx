# PyREx
***Py**thon **R**etrieval of **Ex**oplanet atmospheres*

PyREx is a lightweight Python-based forward model for computing transmission spectra of exoplanetary atmospheres. It utilizes the semi-analytical formalism proposed by Heng & Kitzmann (2017), which provides an efficient method for modeling isothermal, hydrostatic atmospheres.

The model calculates the wavelength-dependent transit radius via an analytical formula which uses data of molecular opacities from pressure and temperature grids, employing bilinear interpolation for cross-sections. It accounts for the scale height and optical depth to determine the modulation of stellar light during a planetary transit.

Our hope is that this simple code can be refied to be fully differentiable for application in ML pipelines. 


For further details on the theoretical framework, refer to:

Heng, K., & Kitzmann, D. (2017). The theory of transmission spectra revisited: a semi-analytical method for interpreting WFC3 data and an unresolved challenge. Monthly Notices of the Royal Astronomical Society, 470(3), 2972–2981. https://doi.org/10.1093/mnras/stx1453
