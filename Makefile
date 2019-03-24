BASE=donatello

TEST_PATH=./

clean-pyc: 
	find . -name '*.pyc' -delete ;
	find . -name '*.pyo' -delete ;
	find . -name '*~' -delete ;

clean-build:
	rm --force --recursive build/ 
	rm --force --recursive dist/ 
	rm --force --recursive *.egg-info

isort:
	sh -c "isort --skip-glob=.tox --recursive . "

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
	docker run $(BASE) -c "pytest $(TEST_PATH)"

test-dev: 
	docker run -v `pwd`:/opt/workspace $(BASE) -c "pytest $(TEST_PATH)"

