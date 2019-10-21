# PyRINEX

A C-based RINEX 3 reading library for Python that does not suck. 
Blazingly-fast, outperforming most, if not all, pure Python implementations, and even C/C++ ones (looking at you, GPSTk).

**Caution**: bleeding-edge, minimally working code may result in difficult to debug bugs. 
Not recommended (although encouraged) for production use.

**What it does not do**: no RINEX 2/ephemeris(SP3) support (yet). I will get to it.

# Installation

Requires C compiler to be present.

    python3 setup.py install

# Usage

See ```tests.py``` for some examples of how to use it.
