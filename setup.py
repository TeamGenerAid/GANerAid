from distutils.core import setup

setup(
    name='GANerAid',
    version='0.1',
    url='https://github.com/TeamGenerAid/GANerAid.git',
    license='MIT License',
    author='TeamGenerAid',
    author_email='generaid.thu@gmail.com',
    description='Gan library to create and validate synthetic tabular data',
    packages=['GANerAid'],
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.2.4',
        'tensorflow>=2.0',
        'pytorch>=1.8.1',
        'scikit-learn>=0.22.2',
        'seaborn>=0.11.1'

    ],
    python_requires='>=3.7',
)
