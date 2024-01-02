var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],  
    style: cytoscapeStyles,
    layout: {
        name: 'grid'
    },
    wheelSensitivity: 0
});

var isScrolling = false;


// Handles the scrolling mechanism
document.getElementById('cy').addEventListener('wheel', function(event) {
    event.preventDefault(); // Prevent the default scroll behavior
    isScrolling = true;

    // Remove any hover effects
    cy.elements().removeClass('show-label faded highlighted-edge');

    var pan = cy.pan();
    var deltaY = event.deltaY;
    var deltaX = event.deltaX;

    // Adjust the pan based on the scroll delta
    cy.pan({
        x: pan.x - deltaX,
        y: pan.y - deltaY
    });

    // Force Cytoscape to update its internal spatial indexing
    //cy.resize();
    
    // Reset the scrolling flag after a delay
    clearTimeout(window.scrollTimeout);
    window.scrollTimeout = setTimeout(function() 
    {
        isScrolling = false;
    }, 200); // Adjust the time as needed
});

// Click action on the node
cy.on('tap', 'node', function(evt){
    var node = evt.target;
    var nodeId = node.id();

    fetch('/processNode/' + nodeId)
        .then(response => response.json())
        .then(data => {
            // Redirect to the visualization page with the calculated value
            window.location.href = '/visualizeRelu/' + nodeId + '?value=' + data.calculatedValue;
        });
});


// Hover over node
cy.on('mouseover', 'node', function(event) {
    if (!isScrolling) {
        var node = event.target;
        
        node.addClass('show-label');
        //node.connectedEdges().addClass('thick');
        //node.addClass('highlighted-node');
        // Select all elements and subtract the current node and its connected edges
        var others = cy.elements().subtract(node).subtract(node.connectedEdges());
        others.addClass('faded');

        console.log(node.classes());
        // // Highlight the connected edges of the hovered node
        // node.connectedEdges().addClass('highlighted-edge');
    }
});

// Mouseout
cy.on('mouseout', 'node', function(event) {
    cy.elements().removeClass('show-label faded highlighted-edge');
    //cy.elements().addClass('thin');
    // event.target.removeClass('show-label');

    // console.log(node.classes());
});


function generateGraph() {
    var startTime = performance.now();

    fetch('/layout', {
        method: 'POST',
        body: new URLSearchParams({
            'numOrangeNodes': document.getElementById('numOrangeNodes').value
        }),
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received data:', data);

        cy.elements().remove();

        var containerWidth = document.getElementById('cy').offsetWidth;
        var containerHeight = document.getElementById('cy').offsetHeight;

        // Determine the number of layers
        var layers = {};
        data.nodes.forEach(node => {
            if (!layers[node.layer]) {
                layers[node.layer] = [];
            }
            layers[node.layer].push(node);
        });

        var layerNames = Object.keys(layers);
        var layerWidth = containerWidth / layerNames.length;

        // Create nodes for each layer
        layerNames.forEach((layerName, layerIndex) => {
            var nodes = layers[layerName];
            var nodeHeight = containerHeight / nodes.length;

            nodes.forEach((node, nodeIndex) => {
                cy.add({
                    group: 'nodes',
                    data: { id: node.id },
                    position: {
                        x: parseInt(layerWidth * layerIndex + layerWidth / 2),
                        y: parseInt(nodeHeight * nodeIndex + nodeHeight / 2)
                    }
                });
            });
        });

        // Add edges
        data.edges.forEach(edge => {
            cy.add({
                group: 'edges',
                data: {
                    id: edge.id,
                    source: edge.source,
                    target: edge.target,
                    lineColor: edge.color,
                    opacity: edge.opacity
                }
            });
        });

        var endTime = performance.now();
        console.log('Frontend rendering duration:', (endTime - startTime) / 1000);
    });
}


function saveGraphState() {
  var elements = cy.json();  // Get the current state of the graph
  localStorage.setItem('cyGraph', JSON.stringify(elements));  // Save it to localStorage
}

document.getElementById('generateGraph').addEventListener('click', generateGraph);

document.getElementById('returnMainNode').addEventListener('click', function () {
  saveGraphState();  // Save state before navigating away
  window.location.href = "/abstractLayout";
});

// Load the graph state when the layout is loaded
document.addEventListener("DOMContentLoaded", function() {
  loadGraphState();
});

function loadGraphState() {
  var saved = localStorage.getItem('cyGraph');
  if (saved) {
    var elements = JSON.parse(saved);
    cy.json(elements);  // Restore the state of the graph
  }
}