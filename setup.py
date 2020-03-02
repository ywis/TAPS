import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name = "TAPS",
	version="0.0.1",
	author="Siqi Liu",
	author_email="sqliu@astro.utoronto.ca",
	description="Stellar Population Fitting package for post-starburst galaxies from the 3DHST survey. The fitting parameters are only age and attenuation in M05, M13 and BC03 models.",
	packages=['TAPS'],
	)