CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Tested on: Windows 11, i7-12650H @ 2.3GHz 16GB, RTX 4060 Laptop GPU

### Introduction
![cornell box result](https://github.com/user-attachments/assets/dbf9f5f3-0717-4621-800b-dd46273e9b3e)

### Glass-like Materials
* If total internal reflection:
    &emsp;&emsp;Reflect wo
    &emsp;&emsp;Set pdf to 1
    &emsp;&emsp;Return reflectance / abs_cos_theta(*wi)
* Else, calculate R uscing Schlick's approximation:
    &emsp;&emsp;If coin_flip(R):
        &emsp;&emsp;&emsp;&emsp;Reflect wo
        &emsp;&emsp;&emsp;&emsp;Set pdf to R
        &emsp;&emsp;&emsp;&emsp;Return R * reflectance / abs_cos_theta(*wi)
    &emsp;&emsp;Else:
        &emsp;&emsp;&emsp;&emsp;Refract wo
        &emsp;&emsp;&emsp;&emsp;Set pdf to 1-R
        &emsp;&emsp;&emsp;&emsp;Return 1-R * transmittance / abs_cos_theta(*wi) / eta^2

### Microfacet Materials
* Cook-Torrance Microfacet Model
$ f_r(\omega_i, \omega_o) = \frac{F(\omega_i, \omega_o) \cdot G(\omega_i, \omega_o) \cdot D(\mathbf{h})}{4 \cdot (\mathbf{n} \cdot \omega_i) \cdot (\mathbf{n} \cdot \omega_o)} $

* Beckmann Normal Distribution Function

$D(\mathbf{h}) = \frac{e^{-\tan^2\theta_m / \alpha^2}}{\pi \alpha^2 \cos^4\theta_m}$,  &emsp;where $\theta_m$ is the angle between $\mathbf{h}$ and $\mathbf{n}$, and $\alpha$ 

* Fresnel with Schlick's approximation
&emsp;
$F(\theta) = f_0 + (1 - f_0)(1 - \cos\theta)^5$

* Smith's Geometry Term

$G(\omega_i, \omega_o) = \frac{1}{1 + \Lambda(\omega_o) + \Lambda(\omega_i)}  \Lambda(\theta) = \frac{-1 + \sqrt{1 + \tan^2\theta}}{2}$ 


 $tan^2\theta = \frac{\alpha^2}{\cos^2\theta}, where\ \alpha\ is\ a\ roughness\ parameter $

### references
[1] https://henryzxu.github.io/pathtracing-p2/

[2] https://youtu.be/gya7x9H3mV0?si=-cCrC2L0buz2hXyx

[3] Microfacet Models for Refraction through Rough Surfaces, EG, 2007


###### TODOs
* Adding BVH 
* Multi Importance Sampling


