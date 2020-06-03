package main

import (

	"fmt"
	"gorgonia.org/gorgonia/examples/mnist"
	"log"
	//"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"time"
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
	
	x := &mat64.Dense{}
	y := &mat64.Dense{}
	var xVal, yVal tensor.Tensor
	
	
	for epoch := 0; epoch < max_epochs; epoch++ {
	
	
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
			
			//gorgonia.Let(x, xVal)
			//gorgonia.Let(y, yVal)
			
			
			//x.bind(xVal)
			//y.bind(yVal)
		
		}
	
	
	}
	
	//Template code to convert Tensor to Dense
	r1 ,c1 := x.Dims()
	r2, c2 := y.Dims()
	xvaldims :=xVal.Shape()
	yvaldims :=yVal.Shape()
	
	data := xVal.Data()
	
	x_mat64 := mat64.NewDense(28, 28, data.([]float64))
		
	r3,c3 := x_mat64.Dims()
	
	fmt.Println("Mat64 Dense Converted :", r3,c3)
	
	fmt.Println(" x size {} {}, y size {} {} ", r1, c1, r2, c2)
	fmt.Println(" xVal size {}{}, yVal size{}{}", xvaldims, yvaldims)

	
	end := time.Now()
	
	fmt.Println("Time taken : ", end.Sub(start))
	fmt.Println(" MNIST sample size : ", num_samples, target_sample)
	

}