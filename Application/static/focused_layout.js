document.addEventListener('DOMContentLoaded', function() 
{   
    // Cy

    var cy = cytoscape({
        container: document.getElementById('cy'),
        style: cytoscapeStyles,
        layout: {
            name: 'grid', // or any other layout that fits your design
            // other layout options
        },
        wheelSensitivity: 0.3,
        zoomingEnabled: false
    });
    
    // import { initializeCustomEvents } from './customEvents.js';
    // initializeCustomEvents(cy);


    var processedData = JSON.parse(localStorage.getItem('processedData'));

    // Remove any existing elements
    cy.elements().remove();

    // Add nodes and edges to the Cytoscape graph
    if (processedData.nodes) {
        processedData.nodes.forEach(node => cy.add(node));
    }
    if (processedData.edges) {
        processedData.edges.forEach(edge => cy.add(edge));
    }
    

    // // Double tap action

    // // Fetch and redirect or perform other actions
    // // Listen for the custom 'doubleTap' event to handle double clicks
    // // Listen for custom 'singleTap' event for focusing on the node
    // cy.on('singleTap', 'node', function(evt) {
    //     var node = evt.target;
    //     // Focus on the node (center and default zoom)
    //     cy.animate({
    //         center: { eles: node },
    //         zoom: 1
    //     }, {
    //         duration: 250
    //     });
    // });

    // // Listen for custom 'doubleTap' event to fetch data and potentially redirect
    // cy.on('doubleTap', 'node', function(evt) {
    //     var node = evt.target;
    //     var nodeId = node.id();
    //     fetch('/api/activation/' + nodeId)
    //         .then(response => response.json())
    //         .then(data => {
    //             if (data.proceed) {
    //                 // Redirect to the visualization page if API allows proceeding
    //                 window.location.href = '/visualizeActivation/' + nodeId;
    //             } else {
    //                 // Handle the situation where no redirection is needed
    //                 console.log(data.message); // Optionally display this message in the UI
    //             }
    //         })
    //         .catch(error => console.error('Error:', error));
    // });


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
            cy.resize();
            
            // Reset the scrolling flag after a delay
            clearTimeout(window.scrollTimeout);
            window.scrollTimeout = setTimeout(function() 
            {
                isScrolling = false;
            }, 200); // Adjust the time as needed
        }
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
        window.location.href = "/abstractLayout";
      });
      


    // Reset the view
    document.getElementById('resetView').addEventListener('click', function () {
        cy.zoom(1); 
        cy.pan({ x: 0, y: 0 }); // Resets pan to the origin (0,0)
    });
});