from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='tripletpairs',

    version='0.2.7',

    description='Package for simulations of triplet-pair states.',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/davidbossanyi/triplet-pair-states/',

    author='David Bossanyi',

    author_email='davebossanyi@gmail.com',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3 :: Only',
        
        'Operating System :: OS Independent'
    ],

    keywords='optical spectroscopy',

    packages=find_packages(exclude=['doc', 'examples']),

    python_requires='>=3.7',

    install_requires=['numpy>=1.18.1','scipy>=1.4.1','matplotlib>=3.1.2'],

)
