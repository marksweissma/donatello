from setuptools import setup, find_packages


REQUIRES = open('package_requirements.txt', 'r').read()


setup(name='donatello',
      version='0.0.16',
      author='Mark Weiss',
      author_email='mark.s.weiss.ma@gmail.com',
      description=open('README.md'),
      url='http://github.com/marksweissma/donatello',
      packages=find_packages(),
      install_requires=REQUIRES,
      summary='ml experimentation framework',
      license='MIT',
      zip_safe=False
      )
