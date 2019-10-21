from setuptools import setup, Extension

module_rinex = Extension('_rinex',
                    define_macros=[('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                    sources=['library.c', 'py_generator.c', 'py_record.c', 'py_sp3.c', 'util.c'])

setup(
    name='rinex',
    version='0.0.2',
    description='Blazing fast RINEX parsing while cutting a few corners',
    author='Nikita Tereshin',
    author_email='nikita.tereshin@gmail.com',
    url='https://github.com/rumkex/python-rinex',
    long_description='''
This is really just a demo package.
''',
    ext_modules=[module_rinex],
    py_modules=['rinex'],
    zip_safe=False
)
