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
	max_epochs := 30
	
	batch_size := 1
	
	num_batches := num_samples/batch_size


	
	if err := inputs.Reshape(num_samples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}	
	
	var xVal, yVal tensor.Tensor
	var x_mat64, y_mat64, yout_mat64 *mat64.Dense
	
	var accuracy int

	var mse_err, LearningRate float64
	
	makeDense := [3]int{28*28,80,10}
	dL_InputLayer := nns.NewDenseLayer( "Input Layer ", "lrelu", makeDense[0], makeDense[1], nil, nil)
	dLHidden      := nns.NewDenseLayer( "Hidden Layer ","lrelu", makeDense[1], makeDense[2], nil, nil)
	LearningRate   = 0.0001
	
	for epoch := 0; epoch < max_epochs; epoch++ {
	
		t_start := time.Now()
		mse_err =0.0
		accuracy=0
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
			
			//r3,c3 = x_mat64.Dims()
			//r4,c4 = y_mat64.Dims()
			//fmt.Println("Input dims: {} {}, output dims {} {}", r3, c3, r4, c4)
			
			nns.SetInput(x_mat64, &dL_InputLayer)
			
			
			
			nns.Forward(x_mat64,dL_InputLayer)  

			nns.LinkLayers(&dL_InputLayer,&dLHidden)
			
			
			yout_mat64 = nns.GetOutput(dL_InputLayer)
			nns.Forward(yout_mat64,dLHidden)  

			nns.OutputDeltaCalc(y_mat64, dLHidden)
			nns.Update(dLHidden, LearningRate)

			nns.Backprop( dL_InputLayer, dLHidden)

			yout_mat64 = nns.GetOutput(dLHidden)
			
			mse_err += nns.Mse(y_mat64, yout_mat64)
			
			nns.Update(dL_InputLayer, LearningRate)
			
			//tmp_mat64 = yout_mat64.T().(mat64.Dense)
			acc, _ := nns.CheckAccuracy(y_mat64, yout_mat64)
			accuracy +=acc
			
			if b%1000 == 0 {
				
				log.Printf("Mse error : %f, Samples : %d, epoc :%d: Accuracy : %f ", mse_err, end,epoch, 100*float64(accuracy)/float64(1000))
				
				//fmt.Println("expected:", y_mat64)
				//fmt.Println("Achieved:", yout_mat64)
				mse_err  =0.0
				accuracy =0.0
				
				
			}
	

		
		}
	
		t_end := time.Now()
		log.Printf("Epoch %d: Time taken : %s :  MSE err: %f", epoch,  t_end.Sub(t_start), mse_err)
	}
	
	
	
	end := time.Now()
	
	fmt.Println("Time taken : ", end.Sub(start))
	fmt.Println(" MNIST sample size : ", num_samples, target_sample)
	
	nns.LogMlpLayer(&dL_InputLayer)
	nns.LogMlpLayer(&dLHidden)
	log.Printf("Learning rate : %f", LearningRate)
	

}