package nns


import (

	//"fmt"
	"math"

)

// - Activation for Tanh 
func ActivationTanh(x float64 ) float64 {


	return math.Tanh(x)

}

// Tanh Derivative function
func DActivationTanh(x float64 ) float64 {

	tanhv := math.Tanh(x)
	return (1 - tanhv*tanhv)

}

func ActivationSigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*x))
}

// ActivationSigmoidDerivative is derivative of sigmoid
func DActivationSigmoid(x float64) float64 {
	sigmoidx := ActivationSigmoid(x)
	return sigmoidx * (1 - sigmoidx)
}


func ActivationReLU(x float64) float64 {

	//avoid function calls
	if x > 0.0 {
		return x
	} 
	
	return 0.0
}

// ActivationSygmoidDerivative is derivative of sigmoid
func DActivationReLU(x float64) float64 {
	
	if( x > 0.0) {
		return  1.0
	} 	
	
	return 0.0
}


func ActivationLReLU(x float64) float64 {

	//avoid function calls
	if x > 0.0 {
		return x
	} 
	
	return 0.01*x
}

// ActivationSygmoidDerivative is derivative of sigmoid
func DActivationLReLU(x float64) float64 {
	
	if( x > 0.0) {
		return  1.0
	} 	
	
	return 0.01
}



