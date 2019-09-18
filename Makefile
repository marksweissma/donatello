BASE=donatello

clean-pyc: 
	find . -name '*.pyc' -delete ;
	find . -name '*.pyo' -delete ;
	find . -name '*~' -delete ;

clean-build:
	rm -rf  build/ 
	rm -rf  dist/ 
	rm -rf  *.egg-info

lint: clean-pyc
	flake8 --exclude=.tox

env: clean-pyc
	docker build -t $(BASE)  .

shell: env
	docker run -it $(BASE) bash

shell-dev: env
	docker run -it -v `pwd`:/opt/workspace $(BASE) bash

run: 
	docker run -it $(BASE) bash

run-dev:
	docker run -it -v `pwd`:/opt/workspace $(BASE) bash

test: env 
	docker run $(BASE) -c "./scripts/run_test.sh"

test-dev: 
	docker run -it -v `pwd`:/opt/workspace $(BASE) -c "./scripts/run_test.sh"

ship-wheel: clean-build
	python setup.py bdist_wheel --universal &&\
    twine upload dist/*

test-wheel: clean-build
	python setup.py bdist_wheel --universal &&\
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
