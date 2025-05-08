#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import sys
import shutil
import glob
import re
import subprocess
import hashlib
from distutils.command.build_ext import build_ext
import setuptools
import onnxruntime as onnxrt

import amct_onnx # pylint: disable=E0401
from amct_onnx.utils.cpp_extension import \
    create_cpp_extension # pylint: disable=E0401
from amct_onnx.utils.cpp_extension import \
    create_cuda_extension
from amct_onnx.utils.cpp_extension import setup # pylint: disable=E0401

from inc.download_inc_files import get_ort_version
from inc.download_inc_files import get_file_urls
from inc.download_inc_files import download_inc_files
from inc.download_inc_files import SUPPORT_ORT_VERSION

CUDA_VERSION_SUPPORT = {
    '1.5.2': '10.2',
    '1.6.0': '10.2',
    '1.8.0': '11.0',
    '1.9.0': '11.4',
    '1.16.0': '11.8'
}
CUD_DIR = os.path.split(os.path.realpath(__file__))[0]


def _find_cuda_home():
    '''Finds the CUDA install path on OS'''
    # 1st try finding cuda in env CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        cuda_home = cuda_home.split(":")[0]
    nvcc = None
    if cuda_home is None:
        try:
            # 2nd try finding cuda according to command nvcc
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'],
                    stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # 3rd try finding cuda according to the default cuda install path
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and os.path.exists(cuda_home):
        print("Using CUDA_HOME={}".format(cuda_home))
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc):
            print("Using nvcc={}".format(nvcc))
        else:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Please set env CUDA_HOME')
    if cuda_home and not os.path.exists(cuda_home):
        print("Can't not find the CUDA install path according to the "
            "CUDA_HOME, please check env CUDA_HOME.")
    return cuda_home, nvcc


CUDA_HOME, NVCC = _find_cuda_home()


def check_onnxrt_version():
    """ check the installed onnxruntime/onnxruntime-gpu version"""
    ort_version = ''
    try:
        import onnxruntime as ort
    except ImportError:
        pass
    else:
        ort_version = ort.__version__

    if ort_version == '':
        raise EnvironmentError(
            "onnxruntime or onnxruntime-gpu is not installed, need to pip "
            "install onnxruntime or onnxruntime-gpu first. "
            "onnxruntime 1.9.0 or onnxruntime-gpu 1.9.0 is suggested.")
    if ort_version not in SUPPORT_ORT_VERSION:
        raise EnvironmentError(
            "amct_onnx now only support onnxruntime/onnxruntime-gpu 1.6.0, "
            "1.8.0 , 1.9.0 and 1.16.0. if need to use opset 13, "
            "onnxruntime/onnxruntime-gpu 1.6.0 is required; "
            "opset 14 requires onnxruntime/onnxruntime-gpu 1.8.0 or 1.9.0; "
            "opset 15 and 16 requires onnxruntime/onnxruntime-gpu 1.16.0")


def get_cuda_version(nvcc):
    """get the version of cuda"""
    version_info = None
    try:
        with open(os.devnull, 'w') as devnull:
            cmd_ret = subprocess.check_output([nvcc, '-V'],
                stderr=devnull).decode().rstrip('\r\n')
            version_info = re.findall(r'\d+\.\d+\.\d+', cmd_ret)[0]
    except Exception:
        version_info = None
    return version_info


def check_cuda_available():
    """ whether the installed onnxruntime is cuda available"""
    import onnxruntime as onnxrt
    ort_version = onnxrt.__version__
    is_cuda_available = False
    if CUDA_HOME and NVCC and \
            'CUDAExecutionProvider' in onnxrt.get_available_providers():
        cuda_version = get_cuda_version(NVCC)
        if cuda_version and cuda_version.startswith(
                CUDA_VERSION_SUPPORT.get(ort_version)):
            is_cuda_available = True
        else:
            raise EnvironmentError(
                "The installed onnxruntime-gpu {} require cuda version {}, "
                "but env CUDA_HOME points to {}; so install the cuda version "
                "onnxruntime required and set env CUDA_HOME to it if you want "
                "to compile cuda verison amct_onnx, or just unset CUDA_HOME "
                "to compile the cpu-only version amct_onnx.".format(
                    ort_version, CUDA_VERSION_SUPPORT.get(ort_version),
                    cuda_version))
    return is_cuda_available


def customize_compiler_for_nvcc(self):
    """inject customized nvcc compile func for .cu files"""
    # extend the compiler with sourc file .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    compile_func = self._compile

    # now redefine the _compile method. change the compiler according to the
    # source file type.
    def _custom_compile(obj, src, ext, cc_args,
                        extra_postargs, pp_opts):
        if src.endswith(".cu"):
            # use the nvcc to compile .cu files
            self.set_executable('compiler_so', NVCC)
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']
        compile_func(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our _custom_compile method into the class
    self._compile = _custom_compile


# run the customize_compiler
class CustomBuildExtension(build_ext):
    """ wrapper for customized build_ext"""
    def build_extensions(self):
        """ wrapper for build_extensions method"""
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def setup_custom(ext_modules):
    """Compile modules """
    setuptools.setup(name='amct_onnx_ops',
                     ext_modules=ext_modules,
                     cmdclass={'build_ext': CustomBuildExtension},
                     zip_safe=False)


def get_src_inc_list(use_cuda):
    """ get the src inc file list"""
    src = [os.path.join(CUD_DIR, 'src/custom_op_library.cpp'),
           os.path.join(CUD_DIR, 'src/ifmr_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/quant_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/dequant_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/ascend_quant_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/ascend_dequant_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/ascend_antiquant_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/dequant_quant.cpp'),
           os.path.join(CUD_DIR, 'src/hfmg_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/search_n_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/search_n_v2_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/dmq_balance_kernel.cpp'),
           os.path.join(CUD_DIR, 'src/amct_utils.cpp'),
           os.path.join(CUD_DIR, 'src/dump_kernel.cpp')]
    inc = [os.path.join(CUD_DIR, 'inc/')]
    if use_cuda:
        src.append(os.path.join(CUD_DIR, 'src/dequant_quant_impl.cu'))
    return src, inc


def try_download_inc_files():
    """ try_download_inc_files"""
    try:
        download_inc_files()
    except Exception as exception:
        print('Download onnxruntime custom_op dependent include files failed, '
              'please download it manually:')
        ort_version = get_ort_version()
        file_urls = get_file_urls(ort_version)
        for url in file_urls.values():
            print(url)
        raise RuntimeError('[ERROR]Automatic download inc files failed.') from exception


def get_tar_lib_name(py_version):
    if sys.hexversion >= 0x030800f0:    # py_version >= 3.8.0
        tar_lib = '^amct_onnx_ops.cpython-{}{}.*so$'.format(py_version.major,
                                                             py_version.minor)
    else:
        tar_lib = '^amct_onnx_ops.cpython-{}{}m.*so$'.format(py_version.major,
                                                             py_version.minor)
    return tar_lib


def add_lib_to_dist_info():
    amct_onnx_ops_lib = os.path.join(amct_onnx.__path__[0],
                                     'custom_op',
                                     'libamct_onnx_ops.so')
    with open(amct_onnx_ops_lib, 'rb') as f:
        sha256obj = hashlib.sha256()
        sha256obj.update(f.read())
        hash_value = sha256obj.hexdigest()
    file_name = os.path.join('amct_onnx', 'custom_op', 'libamct_onnx_ops.so')
    file_size = os.stat(amct_onnx_ops_lib).st_size
    info = file_name + ',sha256=' + hash_value + ',' + str(file_size)
    info_file = os.path.join(os.path.dirname(amct_onnx.__path__[0]),
                             'amct_onnx-{}.dist-info'.format(amct_onnx.__version__),
                             'RECORD')
    with open(info_file, mode='a+') as f:
        f.seek(0)
        lines = f.readlines()
        if lines[-1].split(',')[0] == file_name:
            f.seek(sum(len(line) for line in lines[:-1]) + len(lines) - 1)
            f.truncate()
        f.write(info)
        f.write('\n')


def main(): # pylint: disable=R0914
    """compile amct_onnx_op for amct_onnx tool """
    # check onnxruntime version
    check_onnxrt_version()
    use_cuda = check_cuda_available()
    # download include files from onnxruntime github
    try_download_inc_files()

    onnx_path = os.path.join(amct_onnx.__path__[0], 'custom_op')
    onnx_ops = os.path.join(onnx_path, 'libamct_onnx_ops.so')
    if os.path.exists(onnx_ops):
        os.remove(onnx_ops)

    bulid_path = os.path.join(CUD_DIR, 'build')
    if os.path.exists(bulid_path):
        shutil.rmtree(bulid_path)

    # compile onnx_ops.so
    src, inc = get_src_inc_list(use_cuda)
    if use_cuda:
        extra_compile_args = {'gcc': ["-DUSE_CUDA"],
            'nvcc': ['--ptxas-options=-v', '-c',
                     '--compiler-options', "'-fPIC'"]}
        # libcudart_static is required in 1.16.0 for cuda runtime api
        ort_version = onnxrt.__version__
        if ort_version in ['1.16.0']:
            extra_link_args = ['-lcudart_static']
            library_dirs = [os.path.join(CUDA_HOME, 'lib64')]
        else:
            extra_link_args = []
            library_dirs = []

        module = create_cuda_extension(sources=src, include_dirs=inc,
            extra_compile_args=extra_compile_args, extra_link_args=extra_link_args,
            library_dirs=library_dirs, libraries=[])
        setup_custom([module])
    else:
        module = create_cpp_extension(sources=src, include_dirs=inc,
            extra_compile_args=[], extra_link_args=[],
            library_dirs=[], libraries=[])
        setup([module])
    print('[INFO] Build amct_onnx_op success!')
    # add onnx_ops.so to amct_onnx
    py_version = sys.version_info
    lib_folder = None
    tar_folder = '^lib.*({}.{}|{}{})$'.format(py_version.major, py_version.minor,
                                              py_version.major, py_version.minor)
    for folder in glob.glob('./build/*'):
        if re.findall(tar_folder, folder.split('/')[-1]):
            lib_folder = folder
    if lib_folder is None:
        raise RuntimeError('[ERROR] find lib folder in build failed.')

    amct_onnx_ops_lib = None
    tar_lib = get_tar_lib_name(py_version)
    for file_name in glob.glob('{}/*'.format(lib_folder)):
        if re.findall(tar_lib, file_name.split('/')[-1]):
            amct_onnx_ops_lib = file_name
    if amct_onnx_ops_lib is None:
        raise RuntimeError('[ERROR] find lib in build failed.')

    shutil.copy(amct_onnx_ops_lib, os.path.join(onnx_path, 'libamct_onnx_ops.so'))
    add_lib_to_dist_info()

    print('[INFO] Install amct_onnx_op success!')


if __name__ == '__main__':
    main()
