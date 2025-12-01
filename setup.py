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
	include_package_data=True,
	license="MIT",
	python_requires=">=3.9",
	install_requires=[
		# Kept in sync with requirements.txt
		"pandas>=2.0",
		"numpy>=1.24",
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
