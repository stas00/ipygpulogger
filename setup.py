from setuptools import setup, find_packages

# note: version is maintained inside ipygpulogger/version.py
exec(open('ipygpulogger/version.py').read())

with open("README.md", "r") as fh: long_description = fh.read()

def to_list(buffer): return list(filter(None, map(str.strip, buffer.splitlines())))

requirements = to_list("""
  ipython
  nvidia-ml-py3
  psutil
""")

setup_requirements = ['pytest-runner']

test_requirements = to_list("""
  pytest
  pytest-ipynb
""")

setup(
    name = 'ipygpulogger',
    version = __version__,

    packages = find_packages(include = ['ipygpulogger']),
    include_package_data = True,

    install_requires = requirements,
    setup_requires   = setup_requirements,
    tests_require    = test_requirements,
    python_requires  = '>=3.6',

    test_suite = 'tests',

    license = "Apache License 2.0",

    description = "GPU Logger for jupyter/ipython",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/stas00/ipygpulogger',
    keywords = 'ipygpulogger, jupyter, ipython, memory, gpu',

    author = "Stas Bekman",
    author_email = 'stas@stason.org',

    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    zip_safe = False,
)
