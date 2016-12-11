## MTIL

**M**ulti-**T**ask Feature **I**nteraction **L**earning (MTIL) 


### Publications: 
1. Multi-Task Feature Interaction Learning. </br>
   Kaixiang Lin, Jianpeng Xu, Inci M. Baytas, Shuiwang Ji, Jiayu Zhou</br> 
   **KDD** 2016.  [[Paper]](https://kaixianglin.github.io/papers/2016KDD_MTIL20160924.pdf)
   
### Acknowledgement
This project is based in part upon work supported by the National Science Foundation 
under Grant IIS-1565596 and Office of Naval Research N00014-14-1-0631.


### Folder Structure: </br>
|>MTLconvx: convex formulations </br>
---- MTIL_L_Lc </br>
---- MTIL_L_S </br>
---- MTIL_S_Lc </br>
---- MTIL_S_S </br>

|>MTLnonconvx: nonconvex formulations </br>
---- MTIL_L_Ln </br>
---- MTIL_S_Ln </br>

|>MTLlowrankW: multi-task baseline </br>
---- MTL_L </br>

|>STL_Least: single-task baseline </br>
---- RR </br>
---- STIL </br>

|>cfiles: mex files for accelerating computation </br>

|>datas: synthetic sample data </br>

|>sparsa: [PNOPT solver](https://web.stanford.edu/group/SOL/software/pnopt/) </br>

|>utilities: utilities files for vectorizing matrics, etc.  </br>

script_run_cell_mtl_syn.m   <--- run this to get a sample results.  </br>
configurefile.m             <--- tune parameters here.  </br>


PS:</br>
It's currently the code example for all methods in the paper, we will integrate this to the multi-task learning package [MALSAR](http://malsar.org/) soon.
