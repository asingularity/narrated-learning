python setup.py build_ext --inplace; cp projects/NL/knn_cython.so .
python setup_knn_parallel.py build_ext --inplace
python setup_fast_save_matrix.py build_ext --inplace; cp projects/NL/fast_save_matrix.so .

