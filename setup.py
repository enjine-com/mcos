from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
  name='mcos',
  packages=['mcos'],
  version='0.0.1',
  license='MIT',
  description='Implementation of Monte Carlo Optimization Selection from the paper "A Robust Estimator of the Efficient Frontier"',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='ENJINE',
  author_email='info@enjine.com',
  url='https://github.com/enjine-com/mcos',
  keywords='Monte Carlo convex optimization de-noising clustering shrinkage',
  install_requires=[
    'numpy',
    'pandas'
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Typing :: Typed'
  ],
  python_requires='>=3'
)