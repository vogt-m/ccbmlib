## ccbmlib 1.0 - Conditional Correlated Bernoulli Models for Tanimoto Coefficient Distributions

#### Synopsis
A package to model Tc value distributions for different fingerprints in RDKit.

#### Requirements

The package was built and tested using Python 3.7 and RDKit (2019.03.4).
RDkit has to be installed. It is available from https://rdkit.org.
Apart from that only the popular extensions numpy and scipy are required.

#### How to install

The package is designed to be installed as a Python package using the ``setup.py`` script.

#### Usage

Import the module ``models``:
```Python
import ccbmlib.models as ccbm
```
Set the data directory for storing fingerprints and fingerprint statistics:
```Python
ccbm.set_data_folder("data")
```
This folder will hold the relevant data for building the distribution models and serve as a permanent 
storage space for the fingerprint and statistics files to avoid expensive recalculations.

Fingerprints and statistics are calculated on demand when the corresponding methods are called:
```Python
stats_morgan = ccbm.get_feature_statistics("chembl25_sample", "morgan", {"radius":2}, "chembl25_sample.smi")
stats_maccs = ccbm.get_feature_statistics("chembl25_sample", "maccs", {}, "chembl25_sample.smi")
```
The first parameter is a name identifying the data set. This name is used to store the fingerprint and statistics
information  in the data folder. The second and third parameters define the fingerprint and the (optional) fingerprint
parameters. The final parameter is a filename for the file containing the molecules in Smiles format.
The function returns a ``PairwiseStats`` object from which the distribution models cn be obtained.
Once the statistics have been saved in the data folders, subsequent calls do not need to supply the Smiles file as
the statistics will be retrieved from the data folder
```Python
stats_morgan = ccbm.get_feature_statistics("chembl25_sample", "morgan", {"radius":2})
stats_maccs = ccbm.get_feature_statistics("chembl25_sample", "maccs")
```

The module also contains methods for calculating fingerprints and Tanimoto coefficients, although the will not be as
efficient as using the appropriate RDKit method directly. The following fingerprints are available:

| Name | Method |
| ---- | ------ |
| atom_pairs | ``atom_pairs`` |
| hashed_atom_pairs | ``hashed_atom_pairs`` |
| avalon | ``avalon``|
| maccs | ``maccs_keys`` |
| morgan | ``morgan`` |
| hashed_morgan | ``hashed_morgan`` |
| rdkit | ``rdkit_fingerprint`` |
| torsions | ``torsions`` |
| hashed_torsions | ``hashed_torsions`` |

These methods return fingerprints as lists of features and can be used with the ``tc`` method to calculate the Tanimoto
coefficient. The distribution models are obtained from the stats objects using the``get_tc_distribution`` methods.
The method takes a fingerprint as an optional parameter to obtain the conditional models.

```Python
import rdkit.Chem as Chem
mol1 = Chem.MolFromSmiles("CCN(CC)n1c(=O)c(-c2cn[nH]c2)cc2c(C)nc(N)nc21")
mol2 = Chem.MolFromSmiles("FC(F)(F)c1cc(Br)ncc1-c1nc(N2CCOCC2)nc(N2CCOCC2)n1")

maccs1 = ccbm.maccs_keys(mol1)
maccs2 = ccbm.maccs_keys(mol2)
maccs_tc = ccbm.tc(maccs1,maccs2)
morgan1 = ccbm.morgan(mol1,radius=2)
morgan2 = ccbm.morgan(mol2,radius=2)
morgan_tc = ccbm.tc(maccs1,maccs2)
print("MACCS Tc:",maccs_tc)
maccs_dist = stats_maccs.get_tc_distribution()
print("Significance of MACCS Tc:",maccs_dist.cdf(maccs_tc))
print("Morgan Tc:",maccs_tc)
morgan_dist = stats_morgan.get_tc_distribution()
print("Significance of Morgan Tc:",morgan_dist.cdf(morgan_tc))
print("Significance of the conditional models:")
maccs_cnd = stats_maccs.get_tc_distribution(maccs1)
morgan_cnd = stats_morgan.get_tc_distribution(morgan1)
print("Conditional significance of MACCS Tc:",maccs_cnd.cdf(maccs_tc))
print("Conditional significance of Morgan Tc:",morgan_cnd.cdf(maccs_tc))
```

```
MACCS Tc: 0.5494505494505495
Significance of MACCS Tc: 0.8664363371864638
Morgan Tc: 0.5494505494505495
Significance of Morgan Tc: 0.9999997267443234
Significance of the conditional models:
Conditional significance of MACCS Tc: 0.7112894732958288
Conditional significance of Morgan Tc: 0.9999999992015902
```

Using matplotlib the different distributions can be plotted

```Python
import matplotlib.pyplot as plt
import numpy as np

x_range =  np.arange(0,1.0001,0.01)
p = [maccs_dist.pdf(x) for x in x_range]
c = [maccs_dist.cdf(x) for x in x_range]
plt.plot(x_range,p,'b:')
plt.plot(x_range,c,'b')

p = [maccs_cnd.pdf(x) for x in x_range]
c = [maccs_cnd.cdf(x) for x in x_range]
plt.plot(x_range,p,'b:')
plt.plot(x_range,c,'b')
plt.show()
```
#### Jupyter notebook

The accompanying Jupyter notebook contains code
- for generating statistics data for ChEMBL compounds and
- for generating figures for quality assessment of the distribution models.

#### References

For the theoretical background see the following references;
- Vogt M & Bajorath J.
  Introduction of the Conditional Correlated Bernoulli Model of similarity value distributions and its application
  to the prospective prediction of fingerprint search performance.
  J Chem Inf Model 51, 2496-2506, 2011. https://dx.doi.org/10.1021/ci2003472
- Vogt M & Bajorath J.
  Modeling Tanimoto similarity value distributions and predicting search results.
  Mol Inf 36, 1600131, 2017.  https://doi.org/10.1002/minf.201600131
- Vogt M & Bajorath J.
  ccbmlib – a Python package for modeling Tanimoto similarity value distributions.
  F1000Research 9(Chem Inf Sci), e100, 2020. https://doi.org/10.12688/f1000research.22292.1
  
  

