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
	
	denseLayerName string
	
	
	ActivationFunc           func(v float64) float64
	ActivationDerivative     func(v float64) float64
	
		
}

func     newDenseLayer(layername string, NumInputs int , NumOutputs int, weights []float64) denseLayer {


	var dlayer denseLayer
	
	dlayer.denseLayerName = layername
	dlayer.input  = &mat64.Dense{}
	dlayer.output = &mat64.Dense{}  //Populated during Forward
	
	dlayer.weights = mat64.NewDense(NumInputs, NumOutputs, weights)

	dlayer.biases = mat64.NewDense(1, NumOutputs,nil)  
	
	if weights == nil {
	
		fmt.Println("Input weights null, fill with Random")
		fillRand(dlayer.weights)
	
	}
	
	dlayer.ActivationFunc  = ActivationTanh
	return dlayer

}

func Forward_template( dL *denseLayer) {

	var local *mat64.Dense 
	//var local mat.Dense 

	input   := dL.input
	weights := dL.weights
	
	local.Mul(input, weights)
	
	

	dL.output = local
	//TODO
	//output.Add(local, biases)
	
}

func Forward( input, weights, biases *mat64.Dense, output *mat64.Dense) {

	//var local *mat64.Dense 
	//var local mat.Dense 

	r, c := input.Dims()
	
	fmt.Println("Input Dims:", r, c)
	r2, c2 := weights.Dims()
	fmt.Println("Weights Dims:", r2, c2)
	
	output.Mul(input.T(), weights)
	
	r3, c3 := output.Dims()
	fmt.Println("Output Dims:", r3, c3)
	

	//output = local
	//TODO
	//output.Add(local, biases)
	
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

func NewDense() {

	makeDense := [3]int{10,12,3}
	
	dL_InputLayer := newDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil)

	dLHidden := newDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil)
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
	output := &mat64.Dense{}
	
	
	fillRand(input)
	makeDense := [3]int{10,12,3}
	
	fmt.Println("------------TEST 1 ----------")
	dL_InputLayer := newDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil)
	//dLHidden      := newDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil)
	

	
	
	Forward(input,dL_InputLayer.weights, dL_InputLayer.biases, output)  

	fmt.Println(output)
	

}


