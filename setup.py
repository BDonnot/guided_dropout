from setuptools import setup

setup(name='guidedDropout',
      version='0.1',
      description='Implementation of guided dropout/dropconnect in tensorflow',
      url='https://github.com/BDonnot/GuidedDropout',
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@gmail.com',
      license='GPLv3',
      packages=['GuidedDropout'],
      install_requires=[
          'numpy',
          'tensorflow'
      ],
      zip_safe=False)