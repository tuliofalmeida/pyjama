import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
  
setuptools.setup(

    name="pyjamalib",
    packages = ['pyjamalib'],
    version="0.5.64",
    author="Túlio F. Almeida",
    author_email="tuliofalmeida@hotmail.com",
    description="A library for analyze joint angles from IMU data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuliofalmeida/pyjama",
    install_requires= ['numpy',
                      'pandas',
                      'matplotlib'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)