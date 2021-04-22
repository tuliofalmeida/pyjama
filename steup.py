from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: Production/Stable',
  'Intended Audience :: Researchers, Developers',
  'Operating System :: Windows',
  'License :: MIT License',
  'Programming Language :: Python :: 3.x'
]
 
keyw = [
  'IMU',
  'Euler angle',
  'Kinematics',
  'Quaternion',
  'Joint angle'
]

install = [
  'math >= 1.1.1',
  'numpy >= 1.19.5',
  'matplotlib.pyplot >= 3.2.2',
  'pandas >= 1.1.5',
  'sys',
  'cys',
  'csv >= 1.0'
  'os',
  'time',
  'collections',
  'scipy >= 1.4.1'
]

setup(
  name='pyjama',
  version='0.0.1',
  description='A very basic calculator',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='https://github.com/tuliofalmeida/pyjama',  
  author='Túlio F. Almeida',
  author_email='tuliofalmeida@hotmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords=keyw, 
  packages=find_packages(),
  install_requires=install 
)