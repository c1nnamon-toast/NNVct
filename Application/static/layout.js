var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],  // Initially, there are no elements
    style: [
        {
            selector: 'node',
            style: {
                'label': 'data(id)',
                'overlay-opacity': 0,
            }
        },
        {
            selector: 'node:active',
            style: {
                'overlay-opacity': 0.5,
                'overlay-padding': 4
            }
        },

        {
            selector: 'edge',
            style: {
                'line-color': 'black',
                'line-opacity': 1.0,
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

// Define the styles for faded elements and highlighted edges
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

// Event handler for mouseover on nodes
cy.on('mouseover', 'node', function(event) {
    var node = event.target;
    
    // Select all elements and subtract the current node and its connected edges
    var others = cy.elements().subtract(node).subtract(node.connectedEdges());
    
    // Apply shadowing to everything else
    others.addClass('faded');
    
    // // Highlight the connected edges of the hovered node
    // node.connectedEdges().addClass('highlighted-edge');
});

// Event handler for mouseout
cy.on('mouseout', 'node', function() {
    cy.elements().removeClass('faded highlighted-edge');
});


function generateGraph() {
    // Fetch data from Flask
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
    .then(data => {

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

        // Connect nodes
        for (let i = 0; i < numRedNodes; i++) {
            for (let j = 0; j < numOrangeNodes; j++) {
                let edge = cy.add({
                    group: 'edges',
                    data: {
                        id: 'edge' + i + '-' + j,
                        source: 'redNode' + i,
                        target: 'orangeNode' + j
                    }
                });
                let randomOpacity = 2 * (Math.random() - 0.5);  // Generates a random number between -1 and 1
                if (randomOpacity < 0) {
                    edge.style({
                        'line-color': 'red',
                        'line-opacity': Math.abs(randomOpacity)  // Use the absolute value to ensure opacity is positive
                    });
                } else {
                    edge.style({
                        'line-color': 'green',
                        'line-opacity': randomOpacity
                    });
                }
            }
        }
    });
}


document.getElementById('generateGraph').addEventListener('click', generateGraph);


document.getElementById('returnMainNode').addEventListener('click', function () {
    window.location.href = "/abstract_layout";
});