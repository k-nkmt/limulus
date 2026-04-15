# API Reference

Public API for using limulus from Python.

## Column-Oriented Helpers

`Session` and `DatasetView` expose reshape and derived-column helpers as part of the public API.

```{eval-rst}
.. automethod:: limulus.Session.transpose

.. automethod:: limulus.Session.assign

.. automethod:: limulus.session.DatasetView.transpose

.. automethod:: limulus.session.DatasetView.assign
```

## Top-Level Functions

Convenience functions that can be executed as one-shot operations without creating a session.

```{eval-rst}
.. autofunction:: limulus.submit

.. autofunction:: limulus.run
```

## Session

The main class responsible for dataset management and Data Step execution.

```{eval-rst}
.. autoclass:: limulus.Session
   :members:
   :undoc-members: False

   :exclude-members: select, filter, transpose, assign
```

## DatasetView

A view class returned by :meth:`Session.dataset` for chained operations.

```{eval-rst}
.. autoclass:: limulus.session.DatasetView
   :members:
   :undoc-members: False

   :exclude-members: transpose, assign
```

## SubmitResult

The return value of :meth:`Session.submit`.

```{eval-rst}
.. autoclass:: limulus.SubmitResult
   :members:
```

## LogEntry

A class representing a single entry in the execution log.

```{eval-rst}
.. autoclass:: limulus.LogEntry
   :members:
```
