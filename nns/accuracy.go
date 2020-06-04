package nns


import (

	"fmt"
	"github.com/gonum/matrix/mat64"
)


func MaxIndex(x_mat64 *mat64.Dense) (int, int) {


	var maxIdxR, maxIdxC int
	var maxa float64
	
	maxa = 0.0
	r, c := x_mat64.Dims()
	
	for idxr:=0; idxr <r ; idxr++ {
	
		for idxc:=0; idxc <c ; idxc++ {
	
			if(x_mat64.At(idxr,idxc) > maxa) {
			
				maxa = x_mat64.At(idxr,idxc)
				maxIdxR = idxr
				maxIdxC = idxc
				
			}
		}
	}
	
	return maxIdxR, maxIdxC

}

func CheckAccuracy(actual , estimate *mat64.Dense) (int, int) {

	r1,c1 := actual.Dims()
	r2,c2 := estimate.Dims()

	
	var maxe float64
	var idxr, idxc int
	
	var maxIdxRa, maxIdxCa, maxIdxRe, maxIdxCe int 

	if r1*c1 != r2*c2 {
	
		fmt.Println("Non Fatal : CheckAccuracy : Error - Row/Col mismatch", r1, c1, r2, c2)
		return 0, r1*c1
	}

	maxIdxRa, maxIdxCa = MaxIndex(actual) 
	
	maxe =0.0

	if r1 == c2 {
	
		estimate_64 := estimate.T()
		
		r3,c3 := estimate_64.Dims()
		for idxr=0; idxr <r3 ; idxr++ {
			for idxc=0; idxc <c3 ; idxc++ {
		
				if(estimate_64.At(idxr,idxc) > maxe) {
				
					maxe = estimate_64.At(idxr,idxc)
					maxIdxRe = idxr
					maxIdxCe = idxc
					
				}
			}
		
		}

	} else {
		estimate_64 := estimate
		
		maxIdxRe, maxIdxCe = MaxIndex(estimate_64) 
	
	}

	

	if (maxIdxRa == maxIdxRe) && (maxIdxCa == maxIdxCe) {
		return 1, r1*c1
	} else {
		return 0, r1*c1
	}

}