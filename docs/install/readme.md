---
title: Build from Source
next:
  text: Examples
  link: /examples/mnist
---

# Build from Source

To build from source, you will need cmake and a recent C++ compiler.

```
git clone https://github.com/warg-void/Wolf.git
cd Wolf
cmake -B build -DBUILD_MNIST ON
cmake --build build
```

Then, you can run the examples like so:

```
./build/examples/mnistClassifier
```
That's it!