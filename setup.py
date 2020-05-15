import io
from setuptools import find_packages, setup


setup(name='sentiment_online',
      version='0.1',
      description='sentiment analysis by online learning',
      author='sy',
      author_email='yeop7747@gmail.com',
      packages=find_packages(),
      install_requires=['torch>=0.4.1',
                        'torchvision',
                        'numpy',
                        'pytorch_pretrained_bert',
                        'nltk',
                        'sklearn'],
      zip_safe=False)