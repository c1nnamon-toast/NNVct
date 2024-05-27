document.addEventListener('DOMContentLoaded', function() {
    const nodeId = new URL(window.location.href).pathname.split('/').pop();
    fetchDataAndPlot(nodeId, 0); // Start with 0 or any default value
});

function fetchDataAndPlot(nodeId, nodeValue) {
    fetch(`/api/activation/${nodeId}`)
        .then(response => response.json())
        .then(data => {
            plotActivation(data.weights, data.activation_function, nodeValue);
            displayWeights(data.weights);
        })
        .catch(error => console.error('Error fetching activation data:', error));
}

function plotActivation(weights, activationFunction, nodeValue) {
    if (!activationFunction) {
        activationFunction = "identity";
    }

    var activationFunctions = {
        relu: x => x > 0 ? x : 0,
        sigmoid: x => 1 / (1 + Math.exp(-x)),
        tanh: x => Math.tanh(x),
        identity: x => x,
    };

    var selectedActivationFunction = activationFunctions[activationFunction];
    var trace = {
        x: [],
        y: [],
        mode: 'lines',
        type: 'scatter'
    };
    var highlightTrace = {
        x: [parseFloat(nodeValue)],
        y: [selectedActivationFunction(parseFloat(nodeValue))],
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
        yaxis: { title: 'y' },
        showlegend: false // This line hides the legend
    };

    Plotly.newPlot('ActivationFunctionGraph', data, layout);
}

function displayWeights(weights) {
    const weightsContainer = document.getElementById('weightsDisplay');
    weightsContainer.innerHTML = `<pre>${JSON.stringify(weights, null, 2)}</pre>`;
}

// Function to update the graph with user input
window.updateGraph = function updateGraph() {
    const nodeValue = document.getElementById('nodeValueInput').value || 0; // Use 0 as default
    const nodeId = new URL(window.location.href).pathname.split('/').pop();
    fetchDataAndPlot(nodeId, nodeValue);
}