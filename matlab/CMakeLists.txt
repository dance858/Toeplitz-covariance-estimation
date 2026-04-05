find_package(Matlab REQUIRED)
matlab_add_mex(NAME nml_new_solver SRC nml_new_solver_mex.c LINK_TO nml)
matlab_add_mex(NAME nml_solve SRC nml_solve_mex.c LINK_TO nml)
matlab_add_mex(NAME nml_free_solver SRC nml_free_solver_mex.c LINK_TO nml)
