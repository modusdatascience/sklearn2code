from distutils.core import setup
import versioneer
from setuptools import find_packages

setup(name='sklearn2code',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Serialize scikit-learn estimators',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      packages=find_packages(),
      package_data = {'sklearn2code': ['templates/*']},
      install_requires = ['scikit-learn', 'mako', 'networkx', 'six', 'toolz', 'multipledispatch']
     )