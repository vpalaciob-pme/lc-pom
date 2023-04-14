from setuptools import setup, find_packages

setup(name='LCPOM',
      version='0.0.1',
      description='Python package to simulate Polarized Optical Microscopy images for Liquid Crystal systems',
      long_description=open("README.md", encoding="utf-8").read(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Education',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
      ],
      keywords='Liquid Crystals, Simulations, Optical Microscopy',
      url='https://github.com/depablogroup/lc-pom',
      author='Chuqiao Chen (Elise), Viviana Palacio-Betancur, Pablo Zubieta, Prof. Juan de Pablo',
      author_email='elisechen@uchicago.edu, vpalaciob@uchicago.edu, pzubieta@uchicago.edu, depablo@uchicago.edu',
      license='MIT',
      packages=find_packages(where="src"),
      install_requires=['matplotlib>=3','numpy','scipy'],
      include_package_data=True,
      zip_safe=False)