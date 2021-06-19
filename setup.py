from distutils.core import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GANerAid',
    version='0.2',
    url='https://github.com/TeamGenerAid/GANerAid'
        '',
    license='MIT License',
    author='TeamGenerAid',
    author_email='generaid.thu@gmail.com',
    description='Gan library to create and validate synthetic tabular data',
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.2.4',
        'torch>=1.8.1',
        'scikit-learn>=0.22.2',
        'seaborn>=0.11.1'

    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data = True,
    repository = "https://test.pypi.org/legacy/"
)
