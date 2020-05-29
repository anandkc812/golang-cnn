package nns


import (

	//"fmt"
	"math"

)

// - Activation for Tanh 
func ActivationTanh(v float64 ) float64 {


	return math.Tanh(v)

}

// Tanh Derivative function
func DActivationTanh(v float64 ) float64 {

	tanhv := math.Tanh(v)
	return (1 - tanhv*tanhv)

}

