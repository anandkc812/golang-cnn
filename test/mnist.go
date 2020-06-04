package main

import (

	"fmt"
	"gorgonia.org/gorgonia/examples/mnist"
	"log"
	//"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"time"
	"../nns"
	"github.com/gonum/matrix/mat64"
	
)


const DATA_LOC = "../testdata/mnist/"


type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }



func main()  {

	fmt.Println(" MNIST - Example ")
	var inputs, targets tensor.Tensor
	var err error
	
	start := time.Now()

	// "train" or "test"
	if inputs,targets, err = mnist.Load( "train", DATA_LOC, tensor.Float64); err != nil {

		log.Fatal(err)
	
	}

	num_samples    := inputs.Shape()[0]
	target_sample  := targets.Shape()[0]
	max_epochs := 10
	
	batch_size := 1
	
	num_batches := num_samples/batch_size


	
	if err := inputs.Reshape(num_samples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}	
	
	var xVal, yVal tensor.Tensor
	var x_mat64, y_mat64, yout_mat64 *mat64.Dense
	
	var r3,c3, r4, c4 int

	var mse_err float64
	
	makeDense := [3]int{28*28,80,10}
	dL_InputLayer := nns.NewDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil, nil)
	dLHidden      := nns.NewDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil, nil)

	
	for epoch := 0; epoch < max_epochs; epoch++ {
	
		t_start := time.Now()
		
		for b:= 0; b < num_batches; b++ {
		
			start := b*batch_size
			end   := start+batch_size
			
			if start >= num_samples {
			
				fmt.Println("Error in batch calculation: stat {}, num_samples {} batch_size {}", start, num_samples, batch_size)
				break
			}
			
			if end > num_samples {
				end = num_samples
			}
			
		
			
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}
			
			if yVal, err = targets.Slice(sli{start, end}); err !=nil {
				log.Fatal("Unable to slice y")
			}
			
			if err = xVal.Reshape(batch_size,1,28,28); err != nil {
				log.Fatalf("Unable to reshape epoch", epoch, err )
				
			}
			//data := xVal.Data()
			x_mat64 = mat64.NewDense(28*28, 1, xVal.Data().([]float64))
			y_mat64 = mat64.NewDense(10, 1 , yVal.Data().([]float64))
			
			r3,c3 = x_mat64.Dims()
			r4,c4 = y_mat64.Dims()
			
			
			//fmt.Println("Input dims: {} {}, output dims {} {}", r3, c3, r4, c4)
			
			r3, c3 = nns.SetInput(x_mat64, &dL_InputLayer)
			
			
			
			nns.Forward(x_mat64,dL_InputLayer)  

			nns.LinkLayers(&dL_InputLayer,&dLHidden)
			
			
			yout_mat64 = nns.GetOutput(dL_InputLayer)
			nns.Forward(yout_mat64,dLHidden)  

			nns.OutputDeltaCalc(y_mat64, dLHidden)
			nns.Update(dLHidden, 0.5)

			nns.Backprop( dL_InputLayer, dLHidden)

			yout_mat64 = nns.GetOutput(dLHidden)
			
			mse_err = nns.Mse(y_mat64, yout_mat64)
			
			nns.Update(dL_InputLayer, 0.2)
			
			if b%100 == 0 {
				fmt.Println("Mse error ", mse_err)
			}
	

		
		}
	
		t_end := time.Now()
		fmt.Println("Time taken for epoch {}: time {} mse ", epoch, t_end.Sub(t_start), mse_err)
	}
	
	//Template code to convert Tensor to Dense
	xvaldims :=xVal.Shape()
	yvaldims :=yVal.Shape()
	

	fmt.Println("Mat64 Dense Converted in Loop  :", r3,c3 , r4, c4 )
	fmt.Println(" xVal size {}{}, yVal size{}{}", xvaldims, yvaldims)

	
	end := time.Now()
	
	fmt.Println("Time taken : ", end.Sub(start))
	fmt.Println(" MNIST sample size : ", num_samples, target_sample)
	

}