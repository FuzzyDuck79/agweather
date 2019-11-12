#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools
from Cython.Build import cythonize

if __name__ == '__main__':
    setuptools.setup(
        ext_modules = cythonize('cython/*.pyx',
                                compiler_directives={'language_level': "3"}),
        package_data = {
                        'cyqtree': [
                                    'cython/*'
                                    ]
        }
    )

