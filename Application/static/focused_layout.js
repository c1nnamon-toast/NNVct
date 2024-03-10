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
        window.location.href = "/abstractLayout";
      });
      


    // // Reset the view

    // document.getElementById('resetView').addEventListener('click', function () {
    //     cy.ready(function() {
    //         cy.fit();
    //     });
    // });
});