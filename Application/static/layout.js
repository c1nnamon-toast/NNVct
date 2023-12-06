var cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],  
    style: 
    [
        {
            selector: 'node',
            style: {
                'content': '',
                'overlay-opacity': 0,
                'text-opacity': 1,
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
        },
        {
            selector: '.show-label',
            style: {
                'content': 'data(id)'
            }  
        },

        {
            selector: 'edge',
            style: {
                'line-color': 'data(color)', // Use data attributes for color
                'line-opacity': 'data(opacity)', // Use data attributes for opacity
                // ... other styles for edges ...
            }
        },

        {
            selector: '.highlighted-edge',
            style: {
                'line-color': '#000',
                'line-opacity': 1.0,
                'width': 4 // This sets the content to an empty string
            }
        },
        {
            selector: '.faded',
            style: {
                'opacity': 0.25 // This sets the content to an empty string
            }
        }
    ],
    layout: {
        name: 'grid'
    },
    wheelSensitivity: 0
});

document.getElementById('cy').addEventListener('wheel', function(event) {
    event.preventDefault(); // Prevent the default scroll behavior

    var pan = cy.pan();
    var deltaY = event.deltaY;
    var deltaX = event.deltaX;

    // Adjust the pan based on the scroll delta
    cy.pan({
        x: pan.x - deltaX,
        y: pan.y - deltaY
    });

    cy.resize();
});

// Moseover
cy.on('mouseover', 'node', function(event) {
    var node = event.target;
    
    node.addClass('show-label');
    // Select all elements and subtract the current node and its connected edges
    var others = cy.elements().subtract(node).subtract(node.connectedEdges());
    others.addClass('faded');

    console.log(node.classes());
    // // Highlight the connected edges of the hovered node
    // node.connectedEdges().addClass('highlighted-edge');
});

// Mouseout
cy.on('mouseout', 'node', function(event) {
    cy.elements().removeClass('faded highlighted-edge');
    // event.target.removeClass('show-label');

    // console.log(node.classes());
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
                data: { id: 'redNode' + i},
                position: {
                    x: 0.25 * containerWidth,  // 25% from the left
                    y: spacingRed * (i + 1)  // Space them out vertically
                }
            });
        }

        for (let j = 0; j < numOrangeNodes; j++) {
            cy.add({
                group: 'nodes',
                data: { id: 'orangeNode' + j},
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


function saveGraphState() {
  var elements = cy.json();  // Get the current state of the graph
  localStorage.setItem('cyGraph', JSON.stringify(elements));  // Save it to localStorage
}

document.getElementById('generateGraph').addEventListener('click', generateGraph);

document.getElementById('returnMainNode').addEventListener('click', function () {
  saveGraphState();  // Save state before navigating away
  window.location.href = "/abstract_layout";
});

// Load the graph state when the layout is loaded
document.addEventListener("DOMContentLoaded", function() {
  loadGraphState();
});

function loadGraphState() {
  var saved = localStorage.getItem('cyGraph');
  if (saved) {
    var elements = JSON.parse(saved);
    cy.json(elements);  // Restore the state of the graph
  }
}