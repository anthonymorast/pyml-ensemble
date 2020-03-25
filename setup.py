from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyml_ensemble',
    version='0.1.2',
    description='A library for creating machine learning ensembles.',
    url='https://github.com/anthonymorast/pyml-ensemble',
    author='Anthony Morast',
    author_email='anthony.a.morast@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'scipy'
    ]
)
