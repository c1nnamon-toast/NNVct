var cytoscapeStyles = [
    {
        selector: 'node',
        style: {
            'shape': 'rectangle',
            'background-color': '#AF8FE9',
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
        }
    },
    {
        selector: 'edge',
        style: {
            'width': 2,
            'line-color': '#ccc',
        }
    }
];


// var cytoscapeStyles = 
// [
//     {
//         selector: 'node',
//         style: {
//             'content': '',
//             'overlay-opacity': 0,
//             'text-opacity': 1,
//         },
//     },
//     {
//         selector: 'node',
//         style: {
//             'background-color': 'orange',
//             'overlay-opacity': 0,
//         },
//     },
//     {
//         selector: '.show-label',
//         style: {
//             'content': 'data(id)'
//         }  
//     },

//     {
//         selector: 'edge',
//         style: {
//             'line-color': 'data(lineColor)', // Use data attributes for color
//             'line-opacity': 'data(opacity)', // Use data attributes for opacity
//             // ... other styles for edges ...
//         }
//     },
//     {
//         selector: '.faded',
//         style: {
//             'opacity': 0.12 // This sets the content to an empty string
//         }
//     }
// ];