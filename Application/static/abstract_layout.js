document.addEventListener('DOMContentLoaded', function() {
    const elements = JSON.parse(document.getElementById('elementsData').textContent);

    var cy = cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: cytoscapeStyles, // your defined styles
        layout: {
            name: 'grid', // or any other layout that fits your design
            // other layout options
        }
    });

    cy.on('tap', 'node', function(evt) {
        var clickedLayer = evt.target.id();
        
        fetch('/goToMainLayout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                selectedLayer: clickedLayer
            })
        })
        .then(response => response.json())
        .then(data => {
            cy.elements().remove();
    
            data.nodes.forEach(node => cy.add(node));
            
            data.edges.forEach(edge => cy.add(edge));
        });
    });

    // Additional Cytoscape setup and event listeners
});


