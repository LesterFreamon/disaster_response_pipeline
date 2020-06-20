from distutils.core import setup


with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='disaster_response_pipeline',
    version='0.1.0',
    packages=['src', 'tests'],
    url='',
    license='MIT',
    author='aarnon',
    author_email='',
    description='',
    install_requires=requirements
)
