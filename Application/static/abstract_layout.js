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
        window.location.href = '/layout';
    });

    // Additional Cytoscape setup and event listeners
});