from setuptools import find_packages, setup

print(find_packages())
setup(
  name = 'readability-transformers',      
  packages=find_packages(),
  version = '0.1.6',     
  license='Apache License 2.0',       
  description = 'Package for integrating transformer modules for readability-related NLP tasks.',  
  long_description=open("README.md", "r", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  author = 'Chan Woo Kim',                 
  author_email = 'chanwkim01@gmail.com',    
  url = 'https://github.com/OneTheta/readability-transformers', 
  download_url = 'https://github.com/OneTheta/readability-transformers/archive/refs/tags/0.1.6.zip',    
  python_requires=">=3.6.0",
  keywords = ['transformers'],   
  include_package_data=True,
  install_requires=[          
    'loguru',
    'numpy',
    'pandas',
    'easydict',
    'spacy',
    'torch>=1.0',
    'tqdm',
    'sentence-transformers',
    'nltk',
    'TRUNAJOD',
    'requests',
    'gensim',
    'supar',
],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: Apache Software License",
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)