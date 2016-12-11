

Folder Structure: 
|>MTLconvx: convex formulations
---- MTIL_L_Lc
---- MTIL_L_S
---- MTIL_S_Lc
---- MTIL_S_S

|>MTLnonconvx: nonconvex formulations
---- MTIL_L_Ln
---- MTIL_S_Ln

|>MTLlowrankW: multi-task baseline
---- MTL_L

|>STL_Least: single-task baseline
---- RR
---- STIL

|>cfiles: mex files for accelerating computation

|>datas: synthetic sample data

|>sparsa: [solver](https://web.stanford.edu/group/SOL/software/pnopt/)

|>utilities: utilities files for vectorizing matrics, etc. 

script_run_cell_mtl_syn.m   <--- run this to get a sample results. 
configurefile.m             <--- tune parameters here. 