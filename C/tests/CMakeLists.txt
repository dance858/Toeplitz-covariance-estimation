add_executable(all_tests test_nml.c)
target_include_directories(all_tests PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(all_tests nml)
add_test(NAME AllTests COMMAND all_tests)
