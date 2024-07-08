from setuptools import setup, find_packages

setup(
    name='desc',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'sklearn-extra>=0.2.0',
        'openbabel>=3.1.1',
        'rdkit>=2021.03.1'
    ],
    entry_points={
        'console_scripts': [
            'desc=desc.main:main',
        ],
    },
    author='Albert Masip-Sánchez, Xavier López, Josep M. Poblet',
    author_email='albert.masip@urv.cat, javier.lopez@urv.cat',
    description='An Automated Strategy to Efficiently Account for Dynamic Environmental Effects in Solution',
    url='https://github.com/qcgurv/DESC/tree/main',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: GNU AFFERO GENERAL PUBLIC LICENSE',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
