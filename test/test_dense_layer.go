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


func main() {


	dense_test1()
	nns.NewDense()
	
	fmt.Println("Dense Test 1 completed")
	
	

	
}