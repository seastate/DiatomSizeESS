# DiatomSizeESS
Implementation of the diatom size model used to assess Evolutionarily Stable Strategies in Litchman et al. 2009

Example:
```
from Litchman2009 import diatoms
D = diatoms()
D.setup()
D.seed()
D.solveODEs(n_pers=64,t_mix=64.,n_record=12)
```
To save figures in png format:
```
D.fig1.savefig('example1.png')
D.fig2.savefig('example2.png')
```
To save figures in svg format:
```
D.fig1.savefig('example1.svg')
D.fig2.savefig('example2.svg')
```
