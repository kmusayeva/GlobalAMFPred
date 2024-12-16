from setuptools import setup, find_packages

setup(
    name="mlp",
    version="0.1.0",
    author="Khadija Musayeva",
    author_email="khmusayeva@gmail.com",
    description="Label propagation",
    license="MIT",
    url="https://github.com/kmusayeva/AMF-preds/mlp",
    install_requires=['pandas>=2.0.0', 'matplotlib>=3.7.0', 'numpy>=1.22.0', 'scipy>=1.10.1',
                      'seaborn>=0.13.2', 'scikit-learn>=1.2.0', 'osqp>=0.6.6'
                      ],
    python_requires='>=3.6',
)

