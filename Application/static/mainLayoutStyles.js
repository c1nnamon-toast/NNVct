// Main layout styles 

var cytoscapeStyles = [
    {
        selector: 'node',
        style: {
            'content': '',
            'overlay-opacity': 0,
            'text-opacity': 1,
            'background-color': 'gray',
            'text-valign': 'top',
            'text-halign': 'center',
            'text-margin-y': -4
        },
    },
    {
        selector: 'edge',
        style: {
            'line-fill' : 'linear-gradient',
            'line-gradient-stop-colors': 'data(lineGradient)', // Use data attributes for color
            'line-opacity': 'data(opacity)', // Use data attributes for opacity
        }
    },

    {
        selector: '.show-label',
        style: {
            'content': 'data(id)',
            'font-family': "Impact",
            "font-size": 18
        }  
    },
    {
        selector: '.highlighted-edge',
        style: {
            'width': 4,
        }
    },


    {
        selector: '.faded',
        style: {
            'opacity': 0.12 // This sets the content to an empty string
        }
    },

];