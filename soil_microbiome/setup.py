from setuptools import setup, find_packages

setup(
    name="soil_microbiome",
    version="0.1.0",
    author="Khadija Musayeva",
    author_email="khmusayeva@gmail.com",
    description="Soil microbiome prediction project",
    url="https://github.com/kmusayeva/AMF-preds/soil_microbiome",
    license="MIT",
    install_requires=['pandas>=2.0.0', 'matplotlib>=3.7.0',  'numpy>=1.22.0', 'scipy>=1.10.1',
                      'seaborn>=0.13.2', 'scikit-learn>=1.2.0'
                      ],
    python_requires='>=3.6',
)