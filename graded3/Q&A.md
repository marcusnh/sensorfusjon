Why dont we use the laplace approximation or alternative The Gauss-Newton method?
  - is this because it's an iterated EKF and we are actually using it?


What happens when we decrease R? why does the algorithm go slower?
- Is it because the algorithm will add several landmarks in pose to the same real landmark, making the process slower.
- Or is this because it makes the covariance larger?

What type of errors are we supposed to see in simulated:
- can be very slow with changed varibels in R and Q and alphas?

Notater:
* Usikkerhet i range-bearing ikke gaussisk. Bumerangform, mindre masse-unaturlig selvsikker, jekke seg må den
* Øker R -> økt kovarians rundt landmarks
* Kan øke NIS med å minke usikkerhet i "alt"
* Studass sa det ikke, men ass-algoritmen er treg og kan forbedres med f.eks. kamera. Tror det ble sagt i forelesning
* alpha nummer to i JCBB ble tatt opp til 10^-8. Hvis for liten er det for stor gating, men den kunne tas opp til -8 
* uten at det skulle gå ut over resultat. For mye opp vil gjøre gate for liten
* Den andre alfaen har med "forskyvelse" av målingene å gjøre, mener han sa den ligger i typisk 10^-6 til 10^-3. 
* Kan illustrere punktet over med å tegne tre landmarks og målinger.
* Relevant å snakke om tuning og hvordan ulike verdier endrer resultat og koble til svakheter.
* Bør få kjørt gjenom hele datasettet

Litt om tuning:
* Minket verdiene i sigmas en del for at trajectory skulle bli mer smud
* Se litt mer på hvordan alphaene påvirker konfidensintervall og sånt
