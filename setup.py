
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='dnns',  
     version='0.0.1',
     scripts=['worker.py'] ,
     author="Kevin Ryczko",
     author_email="kryczko@uottawa.ca",
     description="A deep learning package for using HDF5 and Pytorch (Distributed Data Parallel with NVIDIA mixed-precision) with ease.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/CLEANit/dnns.git",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
         "Operating System :: OS Independent",
     ],
 )
