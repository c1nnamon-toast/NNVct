// Main layout styles 

var cytoscapeStyles = [
    {
        selector: 'node',
        style: {
            'content': '',
            'overlay-opacity': 0,
            'text-opacity': 1,
        },
    },
    {
        selector: 'node',
        style: {
            'background-color': 'orange',
            'overlay-opacity': 0,
        },
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
            'line-color': 'data(lineColor)', // Use data attributes for color
            'line-opacity': 'data(opacity)', // Use data attributes for opacity
            // ... other styles for edges ...
        }
    },

    // {
    //     selector: '.highlighted-edge',
    //     style: {
    //         //'line-color': '#000',
    //         //'line-opacity': 1.0,
    //         'width': 5 // This sets the content to an empty string
    //     }
    // },
    // {
    //     selector: '.highlighted-node',
    //     style: {
    //         //'line-color': '#000',
    //         //'line-opacity': 1.0,
    //         'border-style': 'solid' // This sets the content to an empty string
    //     }
    // },
    {
        selector: '.faded',
        style: {
            'opacity': 0.12 // This sets the content to an empty string
        }
    },
    // {
    //     selector: '.thin',
    //     style: {
    //         'width' : 2 // This sets the content to an empty string
    //     }
    // },
    // {
    //     selector: '.thick',
    //     style: {
    //         'width' : 4 // This sets the content to an empty string
    //     }
    // },
];