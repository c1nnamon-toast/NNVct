var cytoscapeStyles = [
    {
        selector: 'node',
        style: {
            'shape': 'rectangle',
            'background-color': '#AF8FE9',
            'label': 'data(label)',
            'text-valign': 'top',
            'text-halign': 'center',
            'width': 'data(width)',
            'height': 'data(height)',
            'text-margin-y': -2
        }
    },
    {
        selector: 'edge',
        style: {
            'width': 5,
            'line-color': '#ccc',
        }
    }
];