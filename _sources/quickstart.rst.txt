===========
Quick Start
===========

Install
=======

Donatello is available on pypi

(currently test pypi until release)

.. code:: bash

   pip install -i https://test.pypi.org/simple/ donatello


This guide starts with a simple classification problem and provides an introduction to donatello

For learning more about how to build modeling graphs or more advanced & power usage features
see subsequent sections and notebooks

Overview
========

Donatello has thre main components

    #. :py:class:`donatello.components.core.Sculpture`
    #. :py:class:`donatello.components.data.Dataset`
    #. :py:class:`donatello.components.estimator.Estimator`

These components leverage other components and the supporting utilties which
can be altered or redefined as needed.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   sculpture
   model_types
