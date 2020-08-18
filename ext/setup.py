from distutils.core import setup
from distutils.extension import Extension
import os
base_path = os.environ.get("BASE_PATH")

ext = Extension(
    "kaldi_model",
    sources=['k3.cc'],
    library_dirs=[f'{base_path}/boost_1_74_0/release/lib', f'{base_path}/gentle/ext/kaldi/tools/OpenBLAS/install/lib'],
    libraries=['boost_python3', 'gfortran', 'm', 'dl', 'pthread', 'openblas'],
    include_dirs=['..', f'{base_path}/gentle/ext/kaldi/src', f'{base_path}/gentle/ext/kaldi/tools/openfst/include',
                  f'{base_path}/gentle/ext/kaldi/tools/OpenBLAS/install/include', f'{base_path}/boost_1_74_0/release/include',
                  '/usr/include/python3.6m'],
    extra_objects=['kaldi/src/online2/libkaldi-online2.so', 'kaldi/src/ivector/libkaldi-ivector.so', 'kaldi/src/nnet3/libkaldi-nnet3.so',
                   'kaldi/src/chain/libkaldi-chain.so', 'kaldi/src/nnet2/libkaldi-nnet2.so', 'kaldi/src/lat/libkaldi-lat.so',
                   'kaldi/src/decoder/libkaldi-decoder.so', 'kaldi/src/cudamatrix/libkaldi-cudamatrix.so', 'kaldi/src/feat/libkaldi-feat.so',
                   'kaldi/src/transform/libkaldi-transform.so', 'kaldi/src/gmm/libkaldi-gmm.so', 'kaldi/src/hmm/libkaldi-hmm.so',
                   'kaldi/src/tree/libkaldi-tree.so', 'kaldi/src/matrix/libkaldi-matrix.so', 'kaldi/src/fstext/libkaldi-fstext.so',
                   'kaldi/src/util/libkaldi-util.so', 'kaldi/src/base/libkaldi-base.so', f'{base_path}/gentle/ext/kaldi/tools/openfst/lib/libfst.so'],
    extra_compile_args=["-O3", "-std=c++11", "-DKALDI_DOUBLEPRECISION=0", "-DHAVE_EXECINFO_H=1",
                        "-DHAVE_CXXABI_H", "-DHAVE_OPENBLAS", "-DNDEBUG", "-rdynamic", "-msse", "-msse2"]
)

setup(
    name='kaldi-model',
    version='1.0',
    ext_modules=[ext])
