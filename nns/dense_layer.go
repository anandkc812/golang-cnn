package nns

import (
    "fmt"
    //"time"
    //"math"
	//"gonum.org/v1/gonum/mat"
	"github.com/gonum/matrix/mat64"
 )





type denseLayer struct {

	input   *mat64.Dense
	output  *mat64.Dense
	
	weights *mat64.Dense
	
	denseLayerName string
	
	
	ActivationFunc           func(v float64) float64
	ActivationDerivative     func(v float64) float64
	
		
}

func     newDenseLayer(numNode int , layername string) denseLayer {


	var dlayer denseLayer
	
	dlayer.denseLayerName = layername
	dlayer.input = &mat64.Dense{}
	
	dlayer.ActivationFunc  = ActivationTanh
	return dlayer

}

func Forward( input, weights, biases *mat64.Dense, output *mat64.Dense) {

	var local *mat64.Dense 
	//var local mat.Dense 

	local.Mul(input, weights)
	
	

	output = local
	//TODO
	//output.Add(local, biases)
	
}


func NewDense() {

	dL := newDenseLayer(10, "Input Layer ")
	
	

	fmt.Println("NewDense init", dL)
}




