package nns



import (

	//"fmt"
	//"math"
	"github.com/gonum/matrix/mat64"
)


func  mse(actual , estimate *mat64.Dense)   float64 {

	tmp := &mat64.Dense{}
	tmp2 := &mat64.Dense{}
	
	//r1,c1 := actual.Dims()
	//fmt.Println("Actual Dims :", r1,c1)
	
	//r2,c2 = estimate.Dims()
	//fmt.Println("Estimate Dims :", r2,c2)
	
	tmp.Sub(actual.T(), estimate)
	
	
	r,c := tmp.Dims()
	//fmt.Println("Tmp dims", r, c)
	
	
	if r == 1 {
		tmp2.Mul(tmp, tmp.T())
	
	} else {
		tmp2.Mul(tmp.T(), tmp)
	}

	mse_err := tmp2.At(0,0)/float64(r*c)
	
	return mse_err
}