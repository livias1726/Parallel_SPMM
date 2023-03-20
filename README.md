# NOTE D'IMPLEMENTAZIONE

## OpenMP

## CUDA
### CSR
* Testata implementazione con k blocchi per ciascun insieme di righe, cosicchè ogni blocco lavori
    su una colonna specifica, come fosse un prodotto matrice vettore --> non supera i 40 GFLOPS