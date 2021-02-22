from setuptools import setup

setup(
    name='rlutils',
    version='0.1',
    packages=['rlutils'],
    license='Apache 2.0',
    description='Reinforcement Learning Utilities',
    entry_points={
        'console_scripts': [
            'rlplot=rlutils.plot:main',
            'rlrun=rlutils.run:main'
        ]
    }
)
