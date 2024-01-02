document.addEventListener('DOMContentLoaded', function() {
    const elements = JSON.parse(document.getElementById('elementsData').textContent);

    const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: cytoscapeStyles, // your defined styles
        layout: {
            name: 'grid', // or any other layout that fits your design
            // other layout options
        }
    });

    // Additional Cytoscape setup and event listeners
});