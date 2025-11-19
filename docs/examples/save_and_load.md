# Saving and Loading Models

Wolf has made it easy to save and load models and Tensors to a file.

## Saving
```cpp
Sequential model(Linear(500, 100), ReLU(), Linear(100, 1));
model.save("output/model");
```

This will save the model layers as well as its weights to a binary file. You are free to specify the file extension however you like.

## Loading
```cpp
auto model = Sequential::load("output/model");
```

There are also save_tensor and load_tensor which works on Tensor objects the same way.