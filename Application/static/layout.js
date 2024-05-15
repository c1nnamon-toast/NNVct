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

import { initializeCustomEvents } from './customEvents.js';
initializeCustomEvents(cy);




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
    
    // Load cached network
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



// Double tap action: Fetch and redirect or perform other actions
// Listen for the custom 'doubleTap' event to handle double clicks
// Listen for custom 'singleTap' event for focusing on the node
cy.on('singleTap', 'node', function(evt) {
    var node = evt.target;
    // Focus on the node (center and default zoom)
    cy.animate({
        center: { eles: node },
        zoom: 1
    }, {
        duration: 250
    });
});

// Listen for custom 'doubleTap' event to fetch data and potentially redirect
cy.on('doubleTap', 'node', function(evt) {
    var node = evt.target;
    var nodeId = node.id();
    fetch('/processNode/' + nodeId)
        .then(response => response.json())
        .then(data => {
            window.location.href = '/visualizeRelu/' + nodeId + '?value=' + data.calculatedValue;
        });
});


// Hover over node

cy.on('mouseover', 'node', function(event) {
    if (!isScrolling) {
        var node = event.target;
        
        var incomingEdges = node.connectedEdges(function(el) {
            return el.target().id() === node.id(); 
        });
        
        var others = cy.elements().subtract(node).subtract(incomingEdges);

        node.addClass('show-label');
        incomingEdges.addClass('highlighted-edge edge-gradient'); 
        
        others.addClass('faded');
    }
});

cy.on('mouseout', 'node', function(event) {
    cy.elements().removeClass('show-label faded highlighted-edge edge-gradient');
    //cy.elements().addClass('thin');
    // event.target.removeClass('show-label');

    // console.log(node.classes());
});



// Return to main node

document.getElementById('returnMainNode').addEventListener('click', function () {
  saveGraphState();  // Save state before navigating away
  window.location.href = "/abstractLayout";
});



// Reset the view
document.getElementById('resetView').addEventListener('click', function () {
    cy.zoom(1); 
    cy.pan({ x: 0, y: 0 }); // Resets pan to the origin (0,0)
});