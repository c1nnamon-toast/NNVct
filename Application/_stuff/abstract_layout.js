document.addEventListener("DOMContentLoaded", function() {
    // Get the height of the container
    var containerHeight = document.getElementById('cy').offsetHeight;

    // Calculate 50% of the container's height
    var nodeSize = 0.01 * containerHeight;

    var cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [
            {
                data: { id: 'mainNode' },
                style: {
                    'background-color': 'blue',
                }
            }
        ],
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(id)',
                    'overlay-opacity': 0,
                    width: 30.0,
                    height: 30.0,
                }
            },
            {
                selector: 'node:active',
                style: {
                    'overlay-opacity': 0.5,  
                    'overlay-color': '#888',  
                    'overlay-padding': 4     
                }
            }
        ],
        layout: {
            name: 'grid'
        }
    });

    cy.on('tap', 'node', function(evt) {
        var node = evt.target;
        if (node.id() === 'mainNode') {
          window.location.href = '/layout';
        }
      });

    loadGraphState();
});

function loadGraphState() {
  var saved = localStorage.getItem('cyGraph');
  if (saved) {
    var elements = JSON.parse(saved);
    // Assume 'cy' is the cytoscape instance for the abstract layout
    cy.json(elements);  // Restore the state of the abstract graph
  }
}
