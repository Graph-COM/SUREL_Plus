from setuptools import setup, Extension
import numpy

__version__ = '2.2'

module = Extension('subg_acc',
                   sources=['subg_acc.c'],
                   extra_compile_args=['-fopenmp'],
                   extra_link_args=['-lgomp'],
                   include_dirs=[numpy.get_include(), 'lib'])

setup(name='SubGAcc',
      version=__version__,
      author='Haoteng Yin',
      author_email='yinht@purdue.edu',
      description='This is an extension library based on C and openmp for accelerating subgraph operations.',
      ext_modules=[module],
      install_requires=['numpy'],
      zip_safe=False,
      url="https://github.com/VeritasYin/subg_acc",
      classifiers=[
           'Programming Language :: Python :: 3',
           'Programming Language :: C',
           'Topic :: Scientific/Engineering',
           'Topic :: Software Development',
           'License :: BSD 2-Clause',
           'Operating System :: Unix',
      ],
      )
