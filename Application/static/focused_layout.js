document.addEventListener('DOMContentLoaded', function() 
{
    var cy = cytoscape({
        container: document.getElementById('cy'),
        layout: {
            name: 'grid', // or any other layout that fits your design
            // other layout options
        },
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
            //cy.resize();
            
            // Reset the scrolling flag after a delay
            clearTimeout(window.scrollTimeout);
            window.scrollTimeout = setTimeout(function() 
            {
                isScrolling = false;
            }, 200); // Adjust the time as needed
        }
    });
});