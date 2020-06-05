package main


import (

	"fmt"
	"math/rand"
	//"gonum.org/v1/gonum/mat"
	"github.com/gonum/matrix/mat64"
	"../nns"
)



func generate(randMatrix [][][]float64) {

	
    for i, innerArray := range randMatrix {
		fmt.Println(innerArray)
        for j := range innerArray[0] {
			for k := range innerArray[1] {
		
				randMatrix[i][j][k] = rand.NormFloat64()
        }
    }
	}
}

func dense_test1()  {

	//dim := [4]int{5,5,3,10} 

	//var mat.dense dnn
	
	//dnn := mat.newDense(dim[0], dim[1], nil)
	dnn := mat64.NewDense(3, 5, nil)
	
	fmt.Println("Dnn: ",dnn)
	//generate(dnn)
	
	fmt.Println(dnn)
	
	//dnn.Forward()
	

}




func test_1() {

	input := mat64.NewDense(10, 1, nil)
	output := mat64.NewDense(3, 1, nil)
	
	fmt.Println("Expected output set begin");
	output.Set(0, 0, 0.9)
	output.Set(1, 0, 0.0)
	output.Set(2, 0, 0.9)
	
	fmt.Println("Expected output set");
	
	for i:=0; i< 10;i++ {
		input.Set(i, 0, 0.1*float64(i))
	}
	
	//nns.fillRand(input)
	makeDense := [3]int{10,12,3}
	
	fmt.Println("------------TEST 1 ----------")
	dL_InputLayer := nns.NewDenseLayer( "Input Layer ", makeDense[0], makeDense[1], nil, nil)
	dLHidden      := nns.NewDenseLayer( "Hidden Layer ", makeDense[1], makeDense[2], nil, nil)
	

	//dL_InputLayer.input = input
	
	_, _ = nns.SetInput(input, &dL_InputLayer)
	
	nns.Forward(input,dL_InputLayer)  
	
	//dLHidden.input = dL_InputLayer.output
	nns.LinkLayers(&dL_InputLayer,&dLHidden)
	
	yout_mat64 := nns.GetOutput(dL_InputLayer)
	nns.Forward(yout_mat64,dLHidden)  


	nns.OutputDeltaCalc(output, dLHidden)
	nns.Update(dLHidden, 0.5)

	nns.Backprop( dL_InputLayer, dLHidden)

	yout_mat64 = nns.GetOutput(dLHidden)
	mse_err := nns.Mse(output, yout_mat64)
	
	nns.Update(dL_InputLayer, 0.5)
	
	fmt.Println("Mse error ", mse_err)
	
	fmt.Println(output)
	

}


func main() {


	//dense_test1()
	test_1()
	//nns.NewDense()
	
	//fmt.Println("Dense Test 1 completed")
	
	

	
}