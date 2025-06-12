

## Physical Units

- Length units are **nanometers (nm)** throughout.
- Simulation box size: $L = 1000 \, \text{nm} = 1 \, \mu m$ 
- Pixel size: 

$$
\text{pixel\_size} = \frac{L}{N_{\text{pixels}}} = \frac{1000 \, \text{nm}}{100} = 10 \, \text{nm/pixel}
$$

- Protein diameter:
$$
d_p = 4 \, \text{nm}
$$

- Protein size in pixels:


$$
d_{dim} = 1
$$
Just let this be one pixel.

## Point Spread Function (PSF)

  

The PSF describes the intensity distribution of a point emitter imaged through a microscope. The lateral profile is approximated by a 2D Gaussian:

  
$$
\text{PSF}_{xy}(x, y; z) = \frac{1}{2 \pi \sigma(z,d_{dim})^2} \exp\left(-\frac{x^2 + y^2}{2 \sigma(z,d_{dim})^2}\right)
$$


  
where $\sigma(z)$ is the standard deviation of the Gaussian at axial position $z$. 

## Lateral PSF Width $\sigma_0$


The lateral resolution $\sigma_0$ (standard deviation at the focal plane $z = z_{\text{focus}}$)  is approximated by. Using the Airy disk: 
$$
\sigma_0 = \frac{0.21 \lambda}{NA}
$$

where:

- $\lambda = 500\, \text{nm}$ (wavelength of light),
- $NA = 1.4$ (numerical aperture of objective lens).
## Axial Dependence of the PSF Width


The PSF width broadens away from the focal plane due to defocusing. The lateral PSF standard deviation as a function of axial displacement $\Delta z = z - z_{\text{focus}}$ is:

$$
\sigma(z,d_{dim}) = \sigma_{psf}(z)+\sigma_{par}(d),
$$
Where, the **Rayleigh formula** for the variance with distance from the focal point,

$$
\sigma_{psf}(z) = \sigma_0 \sqrt{1 + \left( \frac{\Delta z}{z_R} \right)^2}
$$
where $z_R$ is the **Rayleigh range** given by:

$$
z_R = \frac{\pi n \sigma_0^2}{\lambda}
$$

And, 
$$
\sigma_{par}(d) = \frac{d}{2\sqrt{2\log(2)}},
$$

- $n = 1.33$ (refractive index of the medium).
- $\lambda \approx 520 nm$ Wavelength of the light.
## Image Formation (Convolution


The intensity image at time $t$ is formed by placing protein masks at each particle’s 2D position $(x_i, y_i)$ and convolving with the PSF of width $\sigma(z_i)$, where $z_i$ is the axial position. $N$ is the number of particles:

$$
I_t(x,y) = \sum_{i=1}^N \left[ M(x - x_i, y - y_i) * \text{PSF}_{xy}(x, y; z_i) \right]
$$
But since the protein mask is only one pixel we can assume:

$$
I_t(x,y) = \sum_{i=1}^N \text{PSF}_{xy}(x-x_i, y-y_i; z_i)
$$

## Brownian Motion of Particles

Each particle position $\mathbf{r}_i(t) = (x_i(t), y_i(t), z_i(t))$ evolves according to:

$$
\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \sqrt{2 D \Delta t} \boldsymbol{\eta}
$$

where:

- $D$ is the diffusion coefficient. 

- $\boldsymbol{\eta} \sim \mathcal{N}(0, 1)$ is a vector of independent standard normal random variables,

- boundary reflections are applied to keep particles within the box.

  
---

  
## Depth-Dependent Intensity Scaling


To simulate reduced brightness away from the focal plane, an intensity scaling factor depending on axial distance is applied:

$$
S(z_i) = \frac{1}{1 + \frac{z_i}{L/2}}
$$


This creates a smooth decay of brightness as particles move away from focus.

## Poisson Distribution

If the expected number of photons at a pixel is $\mu$, the actual count $I$ is:

$$
I \sim \text{Poisson}(\mu)
$$

This means:

$$
P(I = k) = \frac{\mu^k e^{-\mu}}{k!}
$$

- Photon detection is random.
- The average number of photons is $\mu$.
- The noise (variance) is also $\mu$.
- So, brighter areas are less noisy relative to their intensity.

After computing the ideal intensity image $I_{\text{ideal}}(x, y)$, you add Poisson noise like this:

$$
I_{\text{noisy}}(x, y) \sim \text{Poisson}(I_{\text{ideal}}(x, y) \cdot N_{\text{photons}})
$$

Where $N_{\text{photons}}$ is the total expected photon count per particle.