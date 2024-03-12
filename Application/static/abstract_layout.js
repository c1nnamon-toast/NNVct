document.addEventListener('DOMContentLoaded', function() {
    const elements = JSON.parse(document.getElementById('elementsData').textContent);

    var cy = cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: cytoscapeStyles, // your defined styles
        layout: {
            name: 'grid', // or any other layout that fits your design
        },
        zoomingEnabled: false
    });



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

    
    // Node click

    cy.on('tap', 'node', function(evt) {
        var clickedLayer = evt.target.id();
    
        fetch('/processNodeForFocusedLayout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selectedLayer: clickedLayer,
                width: document.getElementById('cy').offsetWidth, 
                height: document.getElementById('cy').offsetHeight
            })
        })
        .then(response => response.json())
        .then(data => {
            // Store the data in LocalStorage
            localStorage.setItem('processedData', JSON.stringify(data));
            // Redirect to the focusedLayout page
            window.location.href = '/focusedLayout';
        });
    });



    // Load full NN button
    
    document.getElementById('fullNN').addEventListener('click', function () {
        //saveGraphState();  // Save state before navigating away
        window.location.href = "/layout";
    });
    


// // Reset the view
// document.getElementById('resetView').addEventListener('click', function () {
//     cy.fit();
// });
});