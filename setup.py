from setuptools import setup, find_packages


REQUIRES = open('package_requirements.txt', 'r').read()


setup(name='donatello',
      version='NA',
      author='Mark Weiss',
      author_email='mark.s.weiss.ma@gmail.com',
      url='git@github.com/marksweissma/donatello',
      packages=find_packages(),
      install_requires=REQUIRES
      )
