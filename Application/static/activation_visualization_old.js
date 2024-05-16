function relu(x) {
    return x > 0 ? x : 0;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

function plotActivation() {
    // Generate a random value between -10 and 10
    var nodeValue = Math.random() * 20 - 10;

    // Randomly select an activation function
    var activationFunctions = [relu, sigmoid, tanh];
    var activationFunction = activationFunctions[Math.floor(Math.random() * activationFunctions.length)];
    var activationFunctionName = activationFunction.name;

    var trace = {
        x: [],
        y: [],
        mode: 'lines',
        type: 'scatter'
    };

    var highlightTrace = {
        x: [nodeValue],
        y: [activationFunction(nodeValue)],
        mode: 'markers',
        marker: { size: 10, color: 'red' },
        type: 'scatter'
    };

    for (var i = -10; i <= 10; i += 0.5) {
        trace.x.push(i);
        trace.y.push(activationFunction(i));
    }

    var data = [trace, highlightTrace];

    var layout = {
        title: activationFunctionName,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    };

    Plotly.newPlot('ActivationFunctionGraph', data, layout);
}

// Add plotActivation to the global scope
window.plotActivation = plotActivation;

// Plot initial graph
plotActivation();