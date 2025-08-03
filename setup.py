from setuptools import setup, find_packages

setup(
    name='tensortrapx_mh',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'keras-tuner',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
    ],
    author='A Ahsan (HABIB)',
    description='A robust deep learning pipeline for breast cancer prediction with CNN, hyperparameter tuning, and production readiness.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/tensortrapx_mh',  # update later
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

