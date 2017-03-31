from setuptools import setup
from setuptools import find_packages


setup(name='recurrentshop',
      version='1.0.0',
      description='Framework for building complex recurrent neural networks with Keras',
      author='Fariz Rahman',
      author_email='fariz@datalog.ai',
      url='https://github.com/farizrahman4u/recurrentshop',
      download_url='https://github.com/farizrahman4u/recurrentshop',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())
