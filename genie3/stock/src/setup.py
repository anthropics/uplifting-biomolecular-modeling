from setuptools import find_namespace_packages, setup


genie3_packages = find_namespace_packages(where='src')

setup(
      name='genie3',
      version='0.0.1',
      description='de novo protein design through equivariantly diffusing oriented residue clouds',
      package_dir={'': 'src'},
      packages=genie3_packages,
      entry_points={
            'console_scripts': [
                  'genie3=genie3.cli:main',
            ],
      },
      install_requires=[
            'tqdm',
            'numpy>=2.0.2,<3',
            'torch==2.7.1',
            'scipy',
            'wandb',
            'pandas',
            'lightning',
            'biopython<1.86',
            'tensorboard',
            'ml-collections',
            'zstandard',
            'huggingface_hub'
      ],
)
