// Heart

var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],  
    style: cytoscapeStyles,
    layout: {
        name: 'grid'
    },
    wheelSensitivity: 0.3,
    zoomingEnabled: false
});



// Whole Neural Network

document.getElementById('generateGraph').addEventListener('click', generateGraph);

function generateGraph() {
    var startTime = performance.now();

    fetch('/layout', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            width: document.getElementById('cy').offsetWidth, 
            height: document.getElementById('cy').offsetHeight 
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received data:', data);

        cy.elements().remove();

        data.nodes.forEach(node => cy.add(node));
        
        data.edges.forEach(edge => cy.add(edge));

        var endTime = performance.now();
        console.log('Frontend rendering duration:', (endTime - startTime) / 1000);
    });
}

    // Caching
    function saveGraphState() {
        var elements = cy.json();  // Get the current state of the graph
        localStorage.setItem('cyGraph', JSON.stringify(elements));  // Save it to localStorage
    }

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



// Handles the scrolling mechanism

let scrollMode = true; // Default mode is scroll
var isScrolling = false;

document.getElementById('toggleScrollZoom').addEventListener('click', function() 
{
    scrollMode = !scrollMode; // Toggle mode
    if (scrollMode) 
    {
        cy.zoomingEnabled(false);
    } else 
    {
        cy.zoomingEnabled(true);
    }
});

document.getElementById('cy').addEventListener('wheel', function(event) {
    if (scrollMode) 
    {
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
    }
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

cy.on('mouseout', 'node', function(event) {
    cy.elements().removeClass('show-label faded highlighted-edge');
    //cy.elements().addClass('thin');
    // event.target.removeClass('show-label');

    // console.log(node.classes());
});



// Return to main node

document.getElementById('returnMainNode').addEventListener('click', function () {
  saveGraphState();  // Save state before navigating away
  window.location.href = "/abstractLayout";
});



// // Reset the view
// document.getElementById('resetView').addEventListener('click', function () {
//     cy.fit();
// });