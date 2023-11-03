var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],  
    style: 
    [
        {
            selector: 'node',
            style: {
                'label': 'data(id)',
                'overlay-opacity': 0,
            }
        },
        
        {
            selector: 'edg',
            style: {
                'line-color': 'data(color)', // Use data attributes for color
                'line-opacity': 'data(opacity)', // Use data attributes for opacity
                // ... other styles for edges ...
            }
        },

        {
            selector: 'node[id^="redNode"]',
            style: {
                'background-color': 'red',
                'overlay-opacity': 0,
            }
        },

        {
            selector: 'node[id^="orangeNode"]',
            style: {
                'background-color': 'orange',
                'overlay-opacity': 0,
            }
        }
    ],

    layout: {
        name: 'grid'
    }
});

// Highligh edges & fade elements
cy.style()
    .selector('.faded')
    .css({
        'opacity': 0.25
    })
    .selector('.highlighted-edge')
    .css({
        'line-color': '#000',
        'line-opacity': 1.0,
        'width': 4  // Make the line thicker if you want
    })
    .update();

// Moseover
cy.on('mouseover', 'node', function(event) {
    var node = event.target;
    
    // Select all elements and subtract the current node and its connected edges
    var others = cy.elements().subtract(node).subtract(node.connectedEdges());
    others.addClass('faded');
    
    // // Highlight the connected edges of the hovered node
    // node.connectedEdges().addClass('highlighted-edge');
});

// Mouseout
cy.on('mouseout', 'node', function() {
    cy.elements().removeClass('faded highlighted-edge');
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
    .then(data => 
    {  

        cy.elements().remove();

        var containerWidth = document.getElementById('cy').offsetWidth;
        var containerHeight = document.getElementById('cy').offsetHeight;

        var numRedNodes = data.numRedNodes;
        var numOrangeNodes = parseInt(document.getElementById('numOrangeNodes').value) || 1;
        var spacingRed = containerHeight / (numRedNodes + 1);
        var spacingOrange = containerHeight / (numOrangeNodes + 1);

        for (let i = 0; i < numRedNodes; i++) {
            cy.add({
                group: 'nodes',
                data: { id: 'redNode' + i },
                position: {
                    x: 0.25 * containerWidth,  // 25% from the left
                    y: spacingRed * (i + 1)  // Space them out vertically
                }
            });
        }

        for (let j = 0; j < numOrangeNodes; j++) {
            cy.add({
                group: 'nodes',
                data: { id: 'orangeNode' + j },
                position: {
                    x: 0.75 * containerWidth,  // 75% from the left
                    y: spacingOrange * (j + 1)  // Space them out vertically
                }
            });
        }

        // When adding edges, you do not need to specify the style directly
        data.edges.forEach(edgeData => {
            cy.add({
                group: 'edges',
                data: edgeData.data,
                // Do not apply style here
            });
        });

        // Log frontend rendering time
        var endTime = performance.now(); // End frontend timing
        console.log('Frontend rendering duration:', (endTime - startTime) / 1000);

    });

}


document.getElementById('generateGraph').addEventListener('click', generateGraph);

document.getElementById('returnMainNode').addEventListener('click', function () {
    window.location.href = "/abstract_layout";
});