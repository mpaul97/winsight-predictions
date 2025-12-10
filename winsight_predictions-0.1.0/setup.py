from setuptools import setup, find_packages

# Minimal but complete setup using setuptools
setup(
	name="winsight-predictions",
	version="0.1.0",
	description="Predictive features and utilities for sports leagues (NFL/NBA/MLB).",
	long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
	long_description_content_type="text/markdown",
	author="",
	author_email="",
	url="",
	packages=find_packages(include=["winsight_predictions", "winsight_predictions.*"]),
	include_package_data=False,
	license="MIT",
	python_requires=">=3.13",
	install_requires=[
		"pandas>=2.3.3",
		"numpy>=2.3.5",
        "scikit_learn>=1.7.2",
		"mp_sportsipy>=0.6.0",
		"boto3>=1.40.54",
		"joblib>=1.5.2",
		"python-dotenv>=1.2.1",
		"regex>=2025.9.1",
		"setuptools>=72.1.0",
		"tqdm>=4.67.1",
	],
	classifiers=[
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3 :: Only",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Development Status :: 3 - Alpha",
		"Topic :: Software Development :: Libraries",
	],
	project_urls={
		"Source": "",
		"Issues": "",
	},
)
