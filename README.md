# ReQuSim

[![PyPI](http://img.shields.io/pypi/v/requsim.svg)](https://pypi.python.org/pypi/requsim)
[![Docs](https://readthedocs.org/projects/requsim/badge/?version=latest)](https://requsim.readthedocs.io)
[![Tests, Artifacts and Release](https://github.com/jwallnoefer/requsim/actions/workflows/ci.yaml/badge.svg)](https://github.com/jwallnoefer/requsim/actions/workflows/ci.yaml)
[![DOI](https://zenodo.org/badge/413313171.svg)](https://zenodo.org/badge/latestdoi/413313171)


ReQuSim is a simulation platform for quantum repeaters. It allows to evaluate
quantum repeater strategies for long-distance quantum key distribution and
entanglement distribution protocols, while taking into account arbitrary
error models.


## Installation

You can install ReQuSim into your python environment from the Python Package
Index:

```
pip install requsim
```

As with all python packages this can possibly overwrite already installed
package versions in your environment with its dependencies, which is why
installing it in a dedicated virtual environment may be preferable.

## Documentation

The Documentation is hosted on [readthedocs](https://readthedocs.org/) and
includes some example setups of how to use ReQuSim to simulate basic
key distribution protocols.

Documentation: [https://requsim.readthedocs.io](https://requsim.readthedocs.io)

## Scope

The aim of ReQuSim is to model quantum repeater protocols accurately and gain
insight where analytical results are hard to obtain.

The level of abstraction
is chosen such that one can consider very general error models (basically
anything that can be described as a quantum channel), but not modeling down
to the actual physical level.

The abstractions used in ReQuSim lend themselves to describing protocols as
high-level strategies (e.g. if two pairs are present, perform entanglement
swapping), but in principle any strategy can be used to schedule arbitrary
events in the event system.

Classical communication plays an important role in quantum repeater protocols,
and cannot be ignored. Especially, because the timing of when quantum operations
need to be performed for a protocol is the central thing the simulation wants
to capture. ReQuSim allows to take into account the timing information from
classical communication steps, but does not model them down to the level of
individual messages being passed.

In summary, ReQuSim can be used for:
  * Modelling a variety of setups for quantum repeaters, like fiber based and
    free-space based repeater, through flexible loss and noise models.
  * Obtaining numerical key rates for repeater protocols that are challenging to
    evaluate analytically.
  * Testing the effect of strategies for repeater protocols at a high level,
    e.g.
    - Should one discard qubits that sit in storage for too long?
    - Does adding an additional repeater station help for a particular setup?
  * Evaluating the effect of parameters on the overall performance of a
    repeater setup. (e.g. if the error model is based on experimental data,
      this could assist in determining whether improving some experimental
      parameter is worthwhile.)

but it is not intended to:
  * Develop code that directly interacts with future quantum hardware.
  * In detail, model effects at the physical layer and some aspects of link
    layer protocols. (However, they can be incorporated indirectly via quantum
      channels and probability distributions.)
  * Simulate huge networks with 1000s of parties.


There currently is limited support for elementary building blocks other than
Bell pairs (e.g. distribution of GHZ states via a multipartite
repeater architecture),
however, there are no built-in events that support them yet. Extending support
for multipartite states is planned for future versions.


### Other quantum network simulators

ReQuSim has a different scope and aim from some other simulation packages for
quantum networks (list obviously not exhaustive):

  * [SimulaQron](http://www.simulaqron.org/): A distributed classical simulation
    of multiple quantum computers that can use real world classical  
    communication to simulate communication times.
  * [NetSquid](https://netsquid.org/): Includes performance of physical and
    link layer in greater detail. Supports multiple ways to store quantum states
    (e.g. pure states, mixed states, stabilizers).
  * [QuISP](https://github.com/sfc-aqua/quisp): Tracks errors instead of full
    states. While lower level operations are supported, the focus is on
    networking aspects.
  * [QuNetSim](https://github.com/tqsd/QuNetSim): Supports multiple backends
    for simulating quantum objects, which can support lower level operations.
    QuNetSim itself focuses on the networking aspects.

ReQuSim's level of abstraction works very well for exploring and comparing
strategies for quantum repeaters. While it aims to be flexible and
extendable, another set of abstractions might work better for other questions.

## Publications and related projects
An earlier (unreleased) version of requsim was used for this publication:

> Simulating quantum repeater strategies for multiple satellites <br>
> J. Wallnöfer, F. Hahn, M. Gündoğan, J. S. Sidhu, F. Wiesner, N. Walk, J. Eisert, J. Wolters <br>
> Commun Phys **5**, 169 (2022); DOI: [10.1038/s42005-022-00945-9](https://doi.org/10.1038/s42005-022-00945-9) <br>
> Preprint: [arXiv:2110.15806 [quant-ph]](https://doi.org/10.48550/arXiv.2110.15806);
> Code archive: [jwallnoefer/multisat_qrepeater_sim_archive](https://github.com/jwallnoefer/multisat_qrepeater_sim_archive)
