from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

# Get version from version.py
version = {}
with open(os.path.join("courtkeynet", "version.py")) as fp:
    exec(fp.read(), version)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                     '-DPYTHON_EXECUTABLE=' + sys.executable]
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                             cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                             cwd=self.build_temp)

setup(
    name="courtkeynet",
    version=version['__version__'],
    author="Adithya N Raj",
    author_email="adithyanraj03@gmail.com",
    description="badminton court detection model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adithyanraj03/courtkeynet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "albumentations>=0.5.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.50.0",
    ],
    ext_modules=[CMakeExtension('courtkeynet_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
)