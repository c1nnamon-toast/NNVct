document.addEventListener('DOMContentLoaded', function() {
    const nodeId = new URL(window.location.href).pathname.split('/').pop();
    fetchDataAndPlot(nodeId);
});

function fetchDataAndPlot(nodeId) {
    fetch(`/api/activation/${nodeId}`)
        .then(response => response.json())
        .then(data => {
            plotActivation(data.weights, data.activation_function);
        })
        .catch(error => console.error('Error fetching activation data:', error));
}

function plotActivation(weights, activationFunction) {
    var nodeValue = Math.random() * 20 - 10;
    var activationFunctions = {
        relu: x => x > 0 ? x : 0,
        sigmoid: x => 1 / (1 + Math.exp(-x)),
        tanh: x => Math.tanh(x)
    };

    var selectedActivationFunction = activationFunctions[activationFunction];
    var trace = {
        x: [],
        y: [],
        mode: 'lines',
        type: 'scatter'
    };
    var highlightTrace = {
        x: [nodeValue],
        y: [selectedActivationFunction(nodeValue)],
        mode: 'markers',
        marker: { size: 10, color: 'red' },
        type: 'scatter'
    };

    for (var i = -10; i <= 10; i += 0.5) {
        trace.x.push(i);
        trace.y.push(selectedActivationFunction(i));
    }

    var data = [trace, highlightTrace];

    var layout = {
        title: `Activation Function: ${activationFunction}`,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    };

    Plotly.newPlot('ActivationFunctionGraph', data, layout);
}