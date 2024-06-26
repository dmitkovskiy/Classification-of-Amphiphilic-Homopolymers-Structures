# Deep Learning Based Classification of Amphiphilic Homopolymers Structures
## Introduction
The possibility of creating particles with specific surface properties through relatively simple operations has resulted in a growing demand for spherical nanoparticles with grafted polymer chains across numerous scientific and industrial fields. These include the oil industry and the production of nanocomposites, as well as biomedical applications.

In accordance with the aim of the paper, we examined a spherical nanoparticle of radius R, decorated by M macromolecules with N amphiphilic monomer units. The amphiphilic monomer units embody both solvophobic and solvophilic groups and are represented as A-graft-B "dumbbells," consisting of two beads with a diameter σ. The beads A are linked together, forming the chain backbone, while beads B serve as freely rotating side pendants (Fig. 1). The macromolecules are softly grafted onto the nanoparticle surface in the sense that the attachment points - A beads of the first monomer units of grafted macromolecules - are located in a thin ~ σ near-surface layer and are capable of moving freely along the nanoparticle surface. The decorated nanoparticle is immersed in a selective solvent, which is athermal for the main chain A groups and poor for the pendant B groups.

![Fig. 1.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%201.png)

**Figure 1** - Model of an amphiphilic homopolymer with A-graft-B monomer units (a) and a nanoparticle decorated with macromolecules of the amphiphilic homopolymer (b). 

In the article "Geometric Features of Structuring of Amphiphilic Macromolecules on the Surface of a Spherical Nanoparticle", we identified a novel type of ordering of polymer structures [1]. In poor for side-pendant solvent, the macromolecules self-assemble into thin membrane-like ABBA bilayers deviated from spherical nanoparticle. The bilayers form morphological structures, that depend on the grafting density and macromolecular polymerization degree, and can be referred to the classical family of complete embedded minimal surfaces. The plane disk, catenoid, helicoid, Costa and Enneper surfaces (Fig. 2), as well as “double” helicoid and “complex surfaces” (Fig 3) were identified.

![Fig. 2.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%202.png)

**Figure 2** - Typical bilayer structures of amphiphilic homopolymers grafted to a spherical nanoparticle and their corresponding minimal surfaces.

![Fig. 3.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%203.png)

**Figure 3** - Instant snapshots of structures with complex patterns: Double helicoid, Complex minimal surface.

In this repository, we put forth an algorithm for the classification of amphiphilic homopolymers grafted to a spherical nanoparticle, employing an ensemble of CNN models based on flat projections of the original structures.

## Data

The data comprise point clouds of different types of points placed in three-dimensional space. Since some structures can be subdivided into simpler components, the structures were first clustered using the DBSCAN method from the sklearn.cluster module with parameter values eps = 1.6 and min_samples = 13, which yielded the most optimal results. The individual elements were aligned by bringing the inertia tensor to the principal axes. Given that the layers are oriented perpendicularly, a spherical layer of attachment points located near the nanoparticle was considered (Fig. 4).

![Fig. 4.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%204.png)

**Figure 4** - Pre-processing of structure point clouds.

The grafted points were represented as two-dimensional graphs in spherical coordinates (φ, θ) with a size of 24x18 pixels (Fig. 5) and as projections of Cartesian coordinates (x, y) 24x24 pixels (Fig. 6). In all cases, the grafted points exhibited a consistent thickness and exhibited varying shapes. The pixel intensity is determined by the number of beads in the thin spherical layer surrounding the nanoparticle.

![Fig. 5.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%205.png)

**Figure 5** - Pixel images of spherical projections for typical elements in *φ-θ* coordinates.

![Fig. 6.png](https://github.com/dmitkovskiy/Deep-Learning-Based-Classification-of-Amphiphilic-Homopolymers-Structures/raw/main/images/Fig.%206.png)

**Figure 6** - Pixel images of spherical projections for typical elements in *x-y* coordinates.


