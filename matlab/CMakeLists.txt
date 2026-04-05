find_package(Matlab REQUIRED)
matlab_add_mex(NAME NML SRC NML_mex.c LINK_TO nml)
