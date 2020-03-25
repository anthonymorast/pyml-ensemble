from setuptools import setup, find_packages

setup(
    name='pyml_ensemble',
    version='0.1',
    description='A library for creating machine learning ensembles.',
    url='https://github.com/anthonymorast/pyml-ensemble',
    author='Anthony Morast',
    author_email='anthony.a.morast@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'scipy'
    ]
)
