from setuptools import setup, find_packages

setup(
  name = 'x-unet',
  packages = find_packages(exclude=[]),
  version = '0.0.3',
  license='MIT',
  description = 'X-Unet',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/x-unet',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'biomedical segmentation',
    'medical deep learning',
    'unets',
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
