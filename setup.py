from setuptools import setup, find_packages

setup(
	name="multi-dimensional-ai",
	version="0.1.0",
	description="Autonomous AI creature with multi-modal transformer architecture for VR",
	author="Caleb Kirschbaum",
	url="https://github.com/calebhk98/Multi-Dimensional-AI",
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	python_requires=">=3.10",
	install_requires=[
		"torch>=2.0.0",
		"transformers>=4.30.0",
		"einops>=0.7.0",
	],
	extras_require={
		"dev": [
			"pytest>=7.4.0",
			"black>=23.0.0",
			"flake8>=6.0.0",
		],
		"evolution": [
			"deap>=1.4.0",
			"ray[default]>=2.6.0",
		],
		"vr": [
			"pythonosc>=1.8.0",
			"mlagents-envs>=0.30.0",
		],
	},
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3.10",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
	],
)
