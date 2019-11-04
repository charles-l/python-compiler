# A python compiler
*for educational purposes...*

This project is an attempt to build a useful compiler with as few lines of code
as possible and no fewer. The goal is to keep it in the thousands, while still
having non-trivial features such as nice error messages and optimization passes.

Hopefully, it'll still be extensible after the fact so it can be used as a base
for implementing more interesting compiler language features quickly.

* Parser [in progress]
* Type checker/semantic check pass [TODO]
* SSA [TODO]
* Code gen [TODO]
* Optimization passes [TODO]
    * Inlining
    * Vectorizing
    * Dead code removal

### Why Python?

I chose Python for this project, because it's a fairly standard language,
and the syntax is fairly intuitive to everyone (even if they haven't written
Python before). Most compiler resources utilize languages that, frankly are
better suited for compiler implementation (I miss my functional pattern matching
and sum types), but are also more obscure. It takes some time to learn Haskell
or OCaml, and the code isn't as easy for beginners to these languages to grok.
