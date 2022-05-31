window.onload = function(){
    init()
    document.getElementById("form").addEventListener("reset", init);
    document.getElementById("solver").addEventListener("change", (e) => {
        displayChange(e.target.value);
    });
}


function init(){
    var solver = document.getElementById("solver-selectLabel").options[0].value
    displayChange(solver);
}

function displayChange(solver){
    // console.log("solver:", solver);
    if(solver == "adam"){
        document.getElementById("learningRate").style.display = "none";
        document.getElementById("learningRateInit").style.display = "block";
        document.getElementById("powerT").style.display = "none";
        document.getElementById("shuffle").style.display = "block";
        document.getElementById("momentum").style.display = "none";
        document.getElementById("nesterovsMomentum").style.display = "none";
        document.getElementById("earlyStopping").style.display = "block";
        document.getElementById("validationFraction").style.display = "block";
        document.getElementById("beta1").style.display = "block";
        document.getElementById("beta2").style.display = "block";
        document.getElementById("epsilon").style.display = "block";
        document.getElementById("nIterNoChange").style.display = "block";
        document.getElementById("maxFun").style.display = "none";
    }
    else if(solver == "lbfgs"){
        document.getElementById("learningRate").style.display = "none";
        document.getElementById("learningRateInit").style.display = "none";
        document.getElementById("powerT").style.display = "none";
        document.getElementById("shuffle").style.display = "none";
        document.getElementById("momentum").style.display = "none";
        document.getElementById("nesterovsMomentum").style.display = "none";
        document.getElementById("earlyStopping").style.display = "none";
        document.getElementById("validationFraction").style.display = "none";
        document.getElementById("beta1").style.display = "none";
        document.getElementById("beta2").style.display = "none";
        document.getElementById("epsilon").style.display = "none";
        document.getElementById("nIterNoChange").style.display = "none";
        document.getElementById("maxFun").style.display = "block";
    }
    else if(solver == "sgd"){
        document.getElementById("learningRate").style.display = "block";
        document.getElementById("learningRateInit").style.display = "block";
        document.getElementById("powerT").style.display = "block";
        document.getElementById("shuffle").style.display = "block";
        document.getElementById("momentum").style.display = "block";
        document.getElementById("nesterovsMomentum").style.display = "block";
        document.getElementById("earlyStopping").style.display = "block";
        document.getElementById("validationFraction").style.display = "block";
        document.getElementById("beta1").style.display = "none";
        document.getElementById("beta2").style.display = "none";
        document.getElementById("epsilon").style.display = "none";
        document.getElementById("nIterNoChange").style.display = "block";
        document.getElementById("maxFun").style.display = "none";
    }
}