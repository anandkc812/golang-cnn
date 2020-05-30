package nns

import (
    "fmt"
    //"time"
    //"math"
	//"gonum.org/v1/gonum/mat"
	"math/rand"
	"github.com/gonum/matrix/mat64"
 )





type denseLayer struct {

	input   *mat64.Dense
	output  *mat64.Dense
	biases  *mat64.Dense
	
	weights *mat64.Dense
	deltas  *mat64.Dense
	derivatives *mat64.Dense
	
	denseLayerName string
	
	
	ActivationFunc           func(v float64) float64
	DActivationFunc          func(v float64) float64
	
		
}

func     newDenseLayer(layername string, NumInputs int , NumOutputs int, weights []float64, biases []float64) denseLayer {


	var dlayer denseLayer
	
	dlayer.denseLayerName = layername
	dlayer.input  = &mat64.Dense{}
	dlayer.output = &mat64.Dense{}  //Populated during Forward

	dlayer.deltas      = &mat64.Dense{}  //Populated during Forward
	dlayer.derivatives = &mat64.Dense{}  //Output x batch (batch ==1 )

	
	dlayer.weights = mat64.NewDense(NumInputs, NumOutputs, weights)

	dlayer.biases = mat64.NewDense(1, NumOutputs,nil)  
	
	
	
	if weights == nil {
	
		fmt.Println("Input weights null, fill with Random")
		fillRand(dlayer.weights)
	
	}
	
	if biases == nil {
	
		fmt.Println("Input Biases nill, fill with Random")
		fillRand(dlayer.biases)
	
	}
	
	dlayer.ActivationFunc   = ActivationTanh
	dlayer.DActivationFunc  = DActivationTanh
	
	return dlayer

}


//y = f(w*x + b) //(Learn w, and b, with f linear or non-linear activation function)

func Forward( input *mat64.Dense, dlayer denseLayer) {

	//var local *mat64.Dense 
	//var local mat.Dense 

	r1, c1 := dlayer.input.Dims()
	r2, c2 := dlayer.weights.Dims()
	r4, c4 := dlayer.biases.Dims()
	
	fmt.Println("Input Dims:", r1, c1)
	fmt.Println("Weights Dims:", r2, c2)
	fmt.Println("Biases Dims:", r4, c4)

	//Avoids transposes and input vs hidden layer handling
	if r1 == 1 {
		dlayer.output.Mul(input, dlayer.weights)
	} else {
		dlayer.output.Mul(input.T(), dlayer.weights)
	
	}
	
	r3, c3 := dlayer.output.Dims()
	
	
	fmt.Println("Output Dims:", r3, c3)
	
	dlayer.output.Add(dlayer.output, dlayer.biases)

	// Apply non-linear activation function
	dlayer.derivatives.Apply(func(r, c int, v float64) float64 {
		return dlayer.DActivationFunc(v)
      }, dlayer.output)
	
	
  
	dlayer.output.Apply(func(r, c int, v float64) float64 {
  		return dlayer.ActivationFunc(v)
      }, dlayer.output)

}

func fillRand( x *mat64.Dense) {


	r,c := x.Dims()
	
	for i:=0; i < r; i++ {
	
		for j:=0; j < c; j++ {
		
			x.Set(i, j, rand.NormFloat64())
			//fmt.Println(rand.NormFloat64())
		
		}
	
	}

}

func Backprop( input *mat64.Dense, dlayer denseLayer) {


}

// Calculate Delta of the last /output layer
func OutputDeltaCalc( groundtruth_values *mat64.Dense, curr_layer denseLayer) {

	
	r1, c1 := curr_layer.output.Dims()
	r2, c2 := groundtruth_values.Dims()

	fmt.Println("Output dims", r1,c1)
	fmt.Println("groundtruth_values dims", r2,c2)
	

	curr_layer.deltas.Sub(curr_layer.output, groundtruth_values.T())
	
	r3, c3 := curr_layer.deltas.Dims()
	fmt.Println("deltas dims", r3,c3)
	r4, c4 := curr_layer.derivatives.Dims()
	fmt.Println("derivatives dims", r4,c4)
	
	
	
	
	
	curr_layer.deltas.MulElem(curr_layer.deltas, curr_layer.derivatives) 

}


func NewDense() {

	makeDense := [3]int{10,12,3}
	
	dL_InputLayer := newDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil, nil)

	dLHidden := newDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil, nil)
	//dLEnd    := newDenseLayer( "Input Layer ", makeDense[2], makeDense[2], nil)
	
	//var r, c int
	r1, c1 := dL_InputLayer.weights.Dims()
	
	r2, c2 := dLHidden.weights.Dims()

	fmt.Println("Input Layer  Weights dim: ",dL_InputLayer.denseLayerName, r1,c1)
	fmt.Println("Hidden Layer Weights dim: ",dLHidden.denseLayerName, r2,c2)
	
	fmt.Println("NewDense init", dL_InputLayer)
	
	
	test_1()
	
	//fillRand(dL_InputLayer.weights)
	//fmt.Println(dL_InputLayer.weights)
}


func test_1() {

	input := mat64.NewDense(10, 1, nil)
	output := mat64.NewDense(3, 1, nil)
	
	fmt.Println("Expected output set begin");
	output.Set(0, 0, 0.9)
	output.Set(1, 0, 0.0)
	output.Set(2, 0, 0.9)
	
	fmt.Println("Expected output set");
	
	fillRand(input)
	makeDense := [3]int{10,12,3}
	
	fmt.Println("------------TEST 1 ----------")
	dL_InputLayer := newDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil, nil)
	dLHidden      := newDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil, nil)
	

	Forward(input,dL_InputLayer)  
	dLHidden.input = dL_InputLayer.output
	
	Forward(dLHidden.input,dLHidden)  


	OutputDeltaCalc(output, dLHidden)
	
	fmt.Println(output)
	

}


