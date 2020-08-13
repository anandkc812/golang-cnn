# golang-cnn

Current implementation for MLP - Complete.
Supports :
      MLP
      Sigmoid, Tanh, ReLu, LReLu.
      MLP - Forward/Back propagation.

Optimization:
      Most of vector/matrix operation done using GONUM
      Pending : Vectorization - of activation function.

Example:
      test/mnist.go
Usage:
      go.exe run test\mnist.go
      
      MNIST dataset should be in right directory. See test\mnist.go for more details.
      To change activation function, make appropriate changes to the mnist.go file.

Results:
      Comparisions of MLP with various activation function available in
      results/plot_logs_compare.ipynb
      
      Logs related to MLP : Attached in results/ directory

TBD/Work in progress:
      Golang based CNN .      
      Keras Models load /store facility. 
      Sparse /Fully Connected comparisons

