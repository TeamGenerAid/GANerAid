from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='GANerAid',
    version='1.0',

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
        'seaborn>=0.11.1',
        'tqdm>=4.61.1',
        'tab-gan-metrics >= 1.1.4',
        'matplotlib>=3.4.0'


    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['GANerAid'],
    include_package_data = True,
    repository = "https://test.pypi.org/legacy/"
)
