# State-of-the-art snapshot (OpenAlex)

_Generated 2026-06-29 via `docs/sota_openalex.py` against the OpenAlex API._

Counts use title-scoped search (`title.search`): precise but a **lower bound** and a *relative* activity signal, not the absolute field size. Recent window = 2022-present; prior = 2017-2021.

## Frontier activity

| Frontier | Total | Recent | Prior | Trend | In geostat-rs? |
|---|--:|--:|--:|:--:|---|
| Vecchia approximate kriging | 14 | 10 | 4 | up | no (dense LU) |
| Nearest-neighbour GP (NNGP) | 36 | 18 | 16 | flat | no |
| SPDE / GMRF spatial | 46 | 24 | 14 | up | no |
| Multiple-point statistics | 160 | 22 | 48 | down | partial (SGS/SIS) |
| Regression kriging (ML hybrid) | 374 | 77 | 126 | down | yes |
| Random-forest kriging | 29 | 18 | 9 | up | yes (RF+resid) |
| Compositional-data kriging | 20 | 4 | 4 | flat | no |
| Generative geostatistical sim. | 1 | 1 | 0 | new | no (out of scope) |

## Top recent works per frontier

### Vecchia approximate kriging  
`title.search:vecchia approximation`

- **32** cites — Katzfuß et al. (2022), *Scaled Vecchia Approximation for Fast Computer-Model Emulation* — SIAM/ASA Journal on Uncertainty Quantification. doi:[10.1137/20m1352156](https://doi.org/10.1137/20m1352156)
- **6** cites — Zhang et al. (2022), *Multi-Scale Vecchia Approximations of Gaussian Processes* — Journal of Agricultural Biological and Environmental Statistics. doi:[10.1007/s13253-022-00488-0](https://doi.org/10.1007/s13253-022-00488-0)
- **5** cites — Pan et al. (2024), *GPU-Accelerated Vecchia Approximations of Gaussian Processes for Geospatial Data using Batched Matrix Computations* — ?. doi:[10.23919/isc.2024.10528930](https://doi.org/10.23919/isc.2024.10528930)
- **5** cites — Huser et al. (2023), *Vecchia Likelihood Approximation for Accurate and Fast Inference with Intractable Spatial Max-Stable Models* — Journal of Computational and Graphical Statistics. doi:[10.1080/10618600.2023.2285332](https://doi.org/10.1080/10618600.2023.2285332)

### Nearest-neighbour GP (NNGP)  
`title.search:nearest-neighbor gaussian process`

- **131** cites — Weber et al. (2023), *nnSVG for the scalable identification of spatially variable genes using nearest-neighbor Gaussian processes* — Nature Communications. doi:[10.1038/s41467-023-39748-z](https://doi.org/10.1038/s41467-023-39748-z)
- **90** cites — Sumayli (2023), *Development of advanced machine learning models for optimization of methyl ester biofuel production from papaya oil: Gaussian process regression (GPR), multilayer perceptron (MLP), and K-nearest neighbor (KNN) regression models* — Arabian Journal of Chemistry. doi:[10.1016/j.arabjc.2023.104833](https://doi.org/10.1016/j.arabjc.2023.104833)
- **60** cites — Jin et al. (2022), *Optimization and analysis of bioenergy production using machine learning modeling: Multi-layer perceptron, Gaussian processes regression, K-nearest neighbors, and Artificial neural network models* — Energy Reports. doi:[10.1016/j.egyr.2022.10.334](https://doi.org/10.1016/j.egyr.2022.10.334)
- **38** cites — Zhang et al. (2022), *K-Nearest Neighbors Gaussian Process Regression for Urban Radio Map Reconstruction* — IEEE Communications Letters. doi:[10.1109/lcomm.2022.3207210](https://doi.org/10.1109/lcomm.2022.3207210)

### SPDE / GMRF spatial  
`title.search:spde spatial`

- **17** cites — Engel et al. (2022), *Spatial species distribution models: Using Bayes inference with INLA and SPDE to improve the tree species choice for important European tree species* — Forest Ecology and Management. doi:[10.1016/j.foreco.2021.119983](https://doi.org/10.1016/j.foreco.2021.119983)
- **13** cites — Fichera et al. (2023), *Spatial modelling of agro-ecologically significant grassland species using the INLA-SPDE approach* — Scientific Reports. doi:[10.1038/s41598-023-32077-7](https://doi.org/10.1038/s41598-023-32077-7)
- **8** cites — Gacutan et al. (2023), *Assessing drivers of estuarine debris using a Bayesian spatial modelling approach (INLA-SPDE)* — Estuarine Coastal and Shelf Science. doi:[10.1016/j.ecss.2023.108592](https://doi.org/10.1016/j.ecss.2023.108592)
- **7** cites — Hildeman et al. (2022), *Joint spatial modeling of significant wave height and wave period using the SPDE approach* — Probabilistic Engineering Mechanics. doi:[10.1016/j.probengmech.2022.103203](https://doi.org/10.1016/j.probengmech.2022.103203)

### Multiple-point statistics  
`title.search:multiple-point statistics`

- **26** cites — Zuo et al. (2022), *A nearest neighbor multiple-point statistics method for fast geological modeling* — Computers & Geosciences. doi:[10.1016/j.cageo.2022.105208](https://doi.org/10.1016/j.cageo.2022.105208)
- **25** cites — Hou et al. (2023), *Reconstructing Three-dimensional geological structures by the Multiple-point statistics method coupled with a deep neural network: A case study of a metro station in Guangzhou, China* — Tunnelling and Underground Space Technology. doi:[10.1016/j.tust.2023.105089](https://doi.org/10.1016/j.tust.2023.105089)
- **14** cites — Zhou et al. (2023), *Knowledge-based multiple point statistics for soil stratigraphy simulation* — Tunnelling and Underground Space Technology. doi:[10.1016/j.tust.2023.105475](https://doi.org/10.1016/j.tust.2023.105475)
- **8** cites — Fan et al. (2024), *Extraction of weak geochemical anomalies based on multiple-point statistics and local singularity analysis* — Computational Geosciences. doi:[10.1007/s10596-024-10272-3](https://doi.org/10.1007/s10596-024-10272-3)

### Regression kriging (ML hybrid)  
`title.search:regression kriging`

- **92** cites — Takoutsing et al. (2022), *Comparing the prediction performance, uncertainty quantification and extrapolation potential of regression kriging and random forest while accounting for soil measurement errors* — Geoderma. doi:[10.1016/j.geoderma.2022.116192](https://doi.org/10.1016/j.geoderma.2022.116192)
- **44** cites — Zhu et al. (2022), *Digital Mapping of Soil Organic Carbon Based on Machine Learning and Regression Kriging* — Sensors. doi:[10.3390/s22228997](https://doi.org/10.3390/s22228997)
- **30** cites — Jiang et al. (2022), *Above-Ground Biomass Estimation for Coniferous Forests in Northern China Using Regression Kriging and Landsat 9 Images* — Remote Sensing. doi:[10.3390/rs14225734](https://doi.org/10.3390/rs14225734)
- **29** cites — Agyeman et al. (2022), *Prediction of nickel concentration in peri-urban and urban soils using hybridized empirical bayesian kriging and support vector machine regression* — Scientific Reports. doi:[10.1038/s41598-022-06843-y](https://doi.org/10.1038/s41598-022-06843-y)

### Random-forest kriging  
`title.search:random forest kriging`

- **92** cites — Takoutsing et al. (2022), *Comparing the prediction performance, uncertainty quantification and extrapolation potential of regression kriging and random forest while accounting for soil measurement errors* — Geoderma. doi:[10.1016/j.geoderma.2022.116192](https://doi.org/10.1016/j.geoderma.2022.116192)
- **31** cites — Farooq et al. (2022), *Comparison of Random Forest and Kriging Models for Soil Organic Carbon Mapping in the Himalayan Region of Kashmir* — Land. doi:[10.3390/land11122180](https://doi.org/10.3390/land11122180)
- **27** cites — Ho et al. (2024), *Random forest regression kriging modeling for soil organic carbon density estimation using multi-source environmental data in central Vietnamese forests* — Modeling Earth Systems and Environment. doi:[10.1007/s40808-024-02158-1](https://doi.org/10.1007/s40808-024-02158-1)
- **24** cites — Han et al. (2024), *Spatial Prediction of Soil Contaminants Using a Hybrid Random Forest–Ordinary Kriging Model* — Applied Sciences. doi:[10.3390/app14041666](https://doi.org/10.3390/app14041666)

### Compositional-data kriging  
`title.search:compositional kriging`

- **1** cites — Chollett et al. (2025), *Sediment Maps for the Continental Shelf of the US Gulf of America and South Atlantic Bight Using Compositional Kriging* — Geoscience Data Journal. doi:[10.1002/gdj3.70014](https://doi.org/10.1002/gdj3.70014)
- **0** cites — Galushin et al. (2024), *SPATIAL MODELING AND ANALYSIS OF THE CHEMICAL COMPOSITION OF PRECIPITATION BASED ON THE KRIGING METHOD (ON THE EXAMPLE OF THE IRKUTSK REGION)* — Успехи современного естествознания (Advances in Current Natural Sciences). doi:[10.17513/use.38329](https://doi.org/10.17513/use.38329)
- **0** cites — ? (2024), *Prediction of grain size distribution using ordinary kriging and compositional kriging methods* — ARPN Journal of Engineering and Applied Sciences. doi:[10.59018/052476](https://doi.org/10.59018/052476)
- **0** cites — ? (2024), *Compositional kriging analysis: A spatial interpolation method for distributions* — ARPN Journal of Engineering and Applied Sciences. doi:[10.59018/012412](https://doi.org/10.59018/012412)

### Generative geostatistical sim.  
`title.search:generative geostatistical simulation`

- **5** cites — Feng et al. (2025), *Geostatistical Facies Simulation based on Training Image Using Generative Networks and Gradual Deformation* — Mathematical Geosciences. doi:[10.1007/s11004-024-10169-y](https://doi.org/10.1007/s11004-024-10169-y)

---

_Polite-pool contact: gran.huja@gmail.com._
